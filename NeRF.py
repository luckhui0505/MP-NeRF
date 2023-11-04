import torch
import torch.nn as nn
from run_nerf_helpers import *
import os
import imageio
import time

def init_linear_weights(m):
    if isinstance(m, nn.Linear):
        if m.weight.shape[0] in [2, 3]:
            nn.init.xavier_normal_(m.weight, 0.1)
        else:
            nn.init.xavier_normal_(m.weight)
        # nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


class DSKnet(nn.Module):
    def __init__(self, num_img, poses, num_pt, kernel_hwindow, *, random_hwindow=0.25,
                 in_embed=3, random_mode='input', img_embed=32, spatial_embed=0, depth_embed=0,
                 num_hidden=3, num_wide=64, short_cut=False, pattern_init_radius=0.1,
                 isglobal=False, optim_trans=False, optim_spatialvariant_trans=False):
        """
        num_img: number of image, used for deciding the view 。图像数量，用于决定视图的嵌入
        poses: the original poses, used for generating new rays, len(poses) == num_img  ，位姿，用于生成新光线
        num_pt: number of sparse point, we use 5 in the paper，稀疏点数量，文中使用5
        kernel_hwindow: the size of physically equivalent blur kernel, the sparse points are bounded inside the blur kernel. 
                        Can be a very big number，模糊内核大小，稀疏点在模糊核内部有界，可以是一个非常大的数字，表示模糊核最大的窗口数，默认是10
        
        random_hwindow: in training, we randomly perturb the sparse point to model a smooth manifold
                        在训练中，我们随机扰动稀疏点来建模光滑流行随机模式。
        random_mode: 'input' or 'output', it controls whether the random perturb is added to the input of DSK or output of DSK
                    控制随机扰动添加在DSK模块的输入还是输出，
        // the above two parameters do not have big impact on the results，以上两个参数对结果影响不大

        in_embed: embedding for the canonical kernel location 正则核位置的嵌入
        img_embed: the length of the view embedding  视图嵌入的长度
        spatial_embed: embedding for the pixel location of the blur kernel inside an image在图像中模糊核的像素位置的嵌入
        depth_embed: (deprecated) the embedding for the depth of current rays       射线深度的嵌入
        
        num_hidden, num_wide, short_cut: control the structure of the MLP
        pattern_init_radius: the little gain add to the deform location described in Sec. 4.4 增加到第4.4节中描述的变形位置的小增益
        isglobal: control whether the canonical kernel should be shared by all the input views or not, does not have big impact on the results
                     控制规范内核是否应该由所有输入视图共享，对结果没有太大影响，
        optim_trans: whether to optimize the ray origin described in Sec. 4.3
                     是否优化第4.3节中描述的光线原点
        optim_spatialvariant_trans: whether to optimize the ray origin for each view or each kernel point.
                              是否优化每个视图或每个内核点的光线原点
        """
        super().__init__()
        self.t=0
        self.num_pt = num_pt
        self.num_img = num_img
        self.short_cut = short_cut
        self.kernel_hwindow = kernel_hwindow
        self.random_hwindow = random_hwindow  # about 1 pix
        self.random_mode = random_mode
        self.isglobal = isglobal
        # 如果isglobal，则所有视图共用一个模糊核，否则，使用num_img个模糊核，默认是FALSE
        pattern_num = 1 if isglobal else num_img
        assert self.random_mode in ['input', 'output'], f"DSKNet::random_mode {self.random_mode} unrecognized, " \
                                                        f"should be input/output"
        self.register_buffer("poses", poses)
        # 通过register_buffer()登记过的张量：会自动成为模型中的参数，随着模型移动（gpu/cpu）而移动，但是不会随着梯度进行更新。
        # torch.randn返回一个符合均值为0，方差为1的正态分布（标准正态分布）中填充随机数的张量
        self.register_parameter("pattern_pos",
                                nn.Parameter(torch.randn(pattern_num, num_pt, 2)
                                             .type(torch.float32) * pattern_init_radius, True))
        # 改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改
        self.register_parameter("pattern_pos2",
                                nn.Parameter(torch.randn(pattern_num, num_pt, 2)
                                             .type(torch.float32) * pattern_init_radius, True))
        self.register_parameter("pattern_pos3",
                                nn.Parameter(torch.randn(pattern_num, num_pt, 2)
                                             .type(torch.float32) * pattern_init_radius, True))

        # 改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改
        self.optim_trans = optim_trans
        self.optim_sv_trans = optim_spatialvariant_trans

        if optim_trans:
            self.register_parameter("pattern_trans",
                                    nn.Parameter(torch.zeros(pattern_num, num_pt, 2)
                                                 .type(torch.float32), True))

        if in_embed > 0:
            # get_embedder 定义in_embed个单词，维度为2的查询矩阵
            # in_embed为模糊核嵌入的位置，in_embed_fn为查询出的结果，in_embed_cnl查询出的维度
            self.in_embed_fn, self.in_embed_cnl = get_embedder(in_embed, input_dim=2)
        else:
            self.in_embed_fn, self.in_embed_cnl = None, 0

        self.img_embed_cnl = img_embed

        if spatial_embed > 0:
            self.spatial_embed_fn, self.spatial_embed_cnl = get_embedder(spatial_embed, input_dim=2)
        else:
            self.spatial_embed_fn, self.spatial_embed_cnl = None, 0

        if depth_embed > 0:
            self.require_depth = True
            self.depth_embed_fn, self.depth_embed_cnl = get_embedder(depth_embed, input_dim=1)
        else:
            self.require_depth = False
            self.depth_embed_fn, self.depth_embed_cnl = None, 0

        # 核函数维度+视角维度+射线维度+嵌入核函数的位置维度
        in_cnl = self.in_embed_cnl + self.img_embed_cnl + self.depth_embed_cnl + self.spatial_embed_cnl
        # 是否优化每个视图或每个内核点的光线原点，如果优化，则输出还包括光线原点的偏移量
        out_cnl = 1 + 2 + 2 if self.optim_sv_trans else 1 + 2  # u, v, w or u, v, w, dx, dy
        hiddens = [nn.Linear(num_wide, num_wide) if i % 2 == 0 else nn.ReLU()
                   for i in range((num_hidden - 1) * 2)]
        # hiddens = [nn.Linear(num_wide, num_wide), nn.ReLU()] * num_hidden
        self.linears = nn.Sequential(
            nn.Linear(in_cnl, num_wide), nn.ReLU(),
            *hiddens,
        )
        self.linears1 = nn.Sequential(
            nn.Linear((num_wide + in_cnl) if short_cut else num_wide, num_wide), nn.ReLU(),
            nn.Linear(num_wide, out_cnl)
        )
        # 改
        # 参数学习，每条分支系数的学习网络
        self.linears3 = nn.Sequential(
            nn.Linear(in_cnl * 3, 128),nn.SiLU(),
            nn.Linear(128, 3), nn.SiLU(),nn.Softmax(-1)
        )
        # 对linears3网络输出参数从512*5*3变为3维，没有用
        self.linears4 = nn.Sequential(
            nn.Linear(512*5*3, 3), nn.SiLU(), nn.Softmax(-1)
        )
        # 改

        self.linears.apply(init_linear_weights)
        self.linears1.apply(init_linear_weights)
        if img_embed > 0:
            self.register_parameter("img_embed",
                                    nn.Parameter(torch.zeros(num_img, img_embed).type(torch.float32), True))
        else:
            self.img_embed = None

    def forward(self, H, W, K, rays, rays_info):
        """
        inputs: all input has shape (ray_num, cnl)
        outputs: output shape (ray_num, ptnum, 3, 2)  last two dim: [ray_o, ray_d]
        """
        self.t=self.t+1
        # img_embed 视图嵌入长度
        img_idx = rays_info['images_idx'].squeeze(-1)
        img_embed = self.img_embed[img_idx] if self.img_embed is not None else \
            torch.tensor([]).reshape(len(img_idx), self.img_embed_cnl)
        # pattern_pos.expand对其进行维度扩展，-1那一列不需要扩展，只是定义了一个维度
        pt_pos = self.pattern_pos.expand(len(img_idx), -1, -1) if self.isglobal \
            else self.pattern_pos[img_idx]
        # 双曲正切函数tanh的输出范围为(-1，1)
        #  pt_pos表示对所有模糊核的所有五个点的随机数，代表模糊核 kernel
        pt_pos = torch.tanh(pt_pos) * self.kernel_hwindow
        # 对核的随机扰动
        # torch.randn_like（）它返回一个与输入张量大小相同的张量，其中填充了均值为 0 方差为 1 的正态分布的随机值。
        if self.random_hwindow > 0 and self.random_mode == "input":
            random_pos = torch.randn_like(pt_pos) * self.random_hwindow
            pt_pos = pt_pos + random_pos
        # 核位置嵌入
        input_pos = pt_pos  # the first point is the reference point
        if self.in_embed_fn is not None:
            pt_pos = pt_pos * (np.pi / self.kernel_hwindow)
            pt_pos = self.in_embed_fn(pt_pos)

        img_embed_expand = img_embed[:, None].expand(len(img_embed), self.num_pt, self.img_embed_cnl)

        x = torch.cat([pt_pos, img_embed_expand], dim=-1)

        rays_x, rays_y = rays_info['rays_x'], rays_info['rays_y']

        ## 改改改改改
        rays_x2, rays_y2 = rays_info['rays_x'], rays_info['rays_y']
        rays_x3, rays_y3 = rays_info['rays_x'], rays_info['rays_y']
        ## 改改改改改

        # 是否与原图像高宽有相关性，对原图像进行缩放
        if self.spatial_embed_fn is not None:
            spatialx = rays_x / (W / 2 / np.pi) - np.pi
            spatialy = rays_y / (H / 2 / np.pi) - np.pi  # scale 2pi to match the freq in the embedder
            spatial = torch.cat([spatialx, spatialy], dim=-1)
            spatial = self.spatial_embed_fn(spatial)
            spatial = spatial[:, None].expand(len(img_idx), self.num_pt, self.spatial_embed_cnl)
            x = torch.cat([x, spatial], dim=-1)

        if self.depth_embed_fn is not None:
            depth = rays_info['ray_depth']
            depth = depth * np.pi  # TODO: please always check that the depth lies between [0, 1)
            depth = self.depth_embed_fn(depth)
            depth = depth[:, None].expand(len(img_idx), self.num_pt, self.depth_embed_cnl)
            x = torch.cat([x, depth], dim=-1)

        # forward
        x1 = self.linears(x)
        x1 = torch.cat([x, x1], dim=-1) if self.short_cut else x1
        x1 = self.linears1(x1)

        delta_trans = None
        # 是否需要优化光线的起始点，whether to optimize the ray origin for each view or each kernel point
        # delta_pos为第一阶段输出的射线偏移量，delta_trans为原点的位置
        if self.optim_sv_trans:
            delta_trans, delta_pos, weight = torch.split(x1, [2, 2, 1], dim=-1)
        else:
            delta_pos, weight = torch.split(x1, [2, 1], dim=-1)
        # 是否需要优化光线的起始点
        if self.optim_trans:
            delta_trans = self.pattern_trans.expand(len(img_idx), -1, -1) if self.isglobal \
                else self.pattern_trans[img_idx]

        if delta_trans is None:
            delta_trans = torch.zeros_like(delta_pos)

        delta_trans = delta_trans * 0.01
        new_rays_xy = delta_pos + input_pos
        weight = torch.softmax(weight[..., 0], dim=-1)

        if self.random_hwindow > 0 and self.random_mode == 'output':
            raise NotImplementedError(f"{self.random_mode} for self.random_mode is not implemented")

        poses = self.poses[img_idx]
        # get rays from offsetted pt position
        rays_x = (rays_x - K[0, 2] + new_rays_xy[..., 0]) / K[0, 0]
        rays_y = -(rays_y - K[1, 2] + new_rays_xy[..., 1]) / K[1, 1]
        dirs = torch.stack([rays_x - delta_trans[..., 0],
                            rays_y - delta_trans[..., 1],
                            -torch.ones_like(rays_x)], -1)

        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., None, :] * poses[..., None, :3, :3],
                           -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        translation = torch.stack([
            delta_trans[..., 0],
            delta_trans[..., 1],
            torch.zeros_like(rays_x),
            torch.ones_like(rays_x)
        ], dim=-1)
        rays_o = torch.sum(translation[..., None, :] * poses[:, None], dim=-1)
        # rays_o = poses[..., None, :3, -1].expand_as(rays_d)

        align = new_rays_xy[:, 0, :].abs().mean()
        align += (delta_trans[:, 0, :].abs().mean() * 10)

        # 改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改
        pt_pos2 = self.pattern_pos2.expand(len(img_idx), -1, -1) if self.isglobal \
            else self.pattern_pos2[img_idx]
        pt_pos3 = self.pattern_pos3.expand(len(img_idx), -1, -1) if self.isglobal \
            else self.pattern_pos3[img_idx]
        pt_pos2 = torch.tanh(pt_pos2) * self.kernel_hwindow
        pt_pos3 = torch.tanh(pt_pos3) * self.kernel_hwindow
        if self.random_hwindow > 0 and self.random_mode == "input":
            random_pos2 = torch.randn_like(pt_pos2) * self.random_hwindow
            pt_pos2 = pt_pos2 + random_pos2
            random_pos3 = torch.randn_like(pt_pos3) * self.random_hwindow
            pt_pos3 = pt_pos3 + random_pos3
        input_pos2 = pt_pos2  # the first point is the reference point
        input_pos3 = pt_pos3  # the first point is the reference point
        if self.in_embed_fn is not None:
            pt_pos2 = pt_pos2 * (np.pi / self.kernel_hwindow)
            pt_pos2 = self.in_embed_fn(pt_pos2)
            pt_pos3 = pt_pos3 * (np.pi / self.kernel_hwindow)
            pt_pos3 = self.in_embed_fn(pt_pos3)
        x2 = torch.cat([pt_pos2, img_embed_expand], dim=-1)
        x3 = torch.cat([pt_pos3, img_embed_expand], dim=-1)
        if self.spatial_embed_fn is not None:
            spatialx = rays_x2 / (W / 2 / np.pi) - np.pi
            spatialy = rays_y2 / (H / 2 / np.pi) - np.pi  # scale 2pi to match the freq in the embedder
            spatial = torch.cat([spatialx, spatialy], dim=-1)
            spatial = self.spatial_embed_fn(spatial)
            spatial = spatial[:, None].expand(len(img_idx), self.num_pt, self.spatial_embed_cnl)
            x2 = torch.cat([x2, spatial], dim=-1)
            x3 = torch.cat([x3, spatial], dim=-1)

        if self.depth_embed_fn is not None:
            depth = rays_info['ray_depth']
            depth = depth * np.pi  # TODO: please always check that the depth lies between [0, 1)
            depth = self.depth_embed_fn(depth)
            depth = depth[:, None].expand(len(img_idx), self.num_pt, self.depth_embed_cnl)
            x2 = torch.cat([x2, depth], dim=-1)
            x3 = torch.cat([x3, depth], dim=-1)
       
        s = torch.cat([x,x2,x3],dim=-1)
        s= self.linears3(s)
        

        # forward
        x2_1 = self.linears(x2)
        x2_1 = torch.cat([x2, x2_1], dim=-1) if self.short_cut else x2_1
        x2_1 = self.linears1(x2_1)
        x3_1 = self.linears(x3)
        x3_1 = torch.cat([x3, x3_1], dim=-1) if self.short_cut else x3_1
        x3_1 = self.linears1(x3_1)
        delta_trans2 = None
        delta_trans3 = None
        if self.optim_sv_trans:
            delta_trans2, delta_pos2, weight2 = torch.split(x2_1, [2, 2, 1], dim=-1)
            delta_trans3, delta_pos3, weight3 = torch.split(x3_1, [2, 2, 1], dim=-1)
        else:
            delta_pos2, weight2 = torch.split(x2_1, [2, 1], dim=-1)
            delta_pos3, weight3 = torch.split(x3_1, [2, 1], dim=-1)
        # 是否需要优化光线的起始点
        if self.optim_trans:
            delta_trans2 = self.pattern_trans.expand(len(img_idx), -1, -1) if self.isglobal \
                else self.pattern_trans[img_idx]
            delta_trans3 = self.pattern_trans.expand(len(img_idx), -1, -1) if self.isglobal \
                else self.pattern_trans[img_idx]
        if delta_trans2 is None:
            delta_trans2 = torch.zeros_like(delta_pos2)
            delta_trans3 = torch.zeros_like(delta_pos3)

        delta_trans2 = delta_trans2 * 0.01
        delta_trans3 = delta_trans3 * 0.01
        new_rays_xy2 = delta_pos2 + input_pos2
        new_rays_xy3 = delta_pos3 + input_pos3
        weight2 = torch.softmax(weight2[..., 0], dim=-1)
        weight3 = torch.softmax(weight3[..., 0], dim=-1)

        rays_x2 = (rays_x2 - K[0, 2] + new_rays_xy2[..., 0]) / K[0, 0]
        rays_y2 = -(rays_y2 - K[1, 2] + new_rays_xy2[..., 1]) / K[1, 1]
        rays_x3 = (rays_x3 - K[0, 2] + new_rays_xy3[..., 0]) / K[0, 0]
        rays_y3 = -(rays_y3 - K[1, 2] + new_rays_xy3[..., 1]) / K[1, 1]
        dirs2 = torch.stack([rays_x2 - delta_trans2[..., 0],
                            rays_y2 - delta_trans2[..., 1],
                            -torch.ones_like(rays_x2)], -1)
        dirs3 = torch.stack([rays_x3 - delta_trans3[..., 0],
                            rays_y3 - delta_trans3[..., 1],
                            -torch.ones_like(rays_x3)], -1)
        rays_d2 = torch.sum(dirs2[..., None, :] * poses[..., None, :3, :3],
                           -1)
        rays_d3 = torch.sum(dirs3[..., None, :] * poses[..., None, :3, :3],
                           -1)
        translation2 = torch.stack([
            delta_trans2[..., 0],
            delta_trans2[..., 1],
            torch.zeros_like(rays_x2),
            torch.ones_like(rays_x2)
        ], dim=-1)
        translation3 = torch.stack([
            delta_trans3[..., 0],
            delta_trans3[..., 1],
            torch.zeros_like(rays_x3),
            torch.ones_like(rays_x3)
        ], dim=-1)
        rays_o2 = torch.sum(translation2[..., None, :] * poses[:, None], dim=-1)
        rays_o3 = torch.sum(translation3[..., None, :] * poses[:, None], dim=-1)

        align2 = new_rays_xy2[:, 0, :].abs().mean()
        align2 += (delta_trans2[:, 0, :].abs().mean() * 10)


        align3 = new_rays_xy3[:, 0, :].abs().mean()
        align3 += (delta_trans3[:, 0, :].abs().mean() * 10)

        # 改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改改
        # s 的维度是512,5,3,512是N_rand ，batch_size
        # s=s.reshape(1,-1)
        # s=self.linears4(s)
        s=s.permute([2,1,0])
        pool=nn.AdaptiveAvgPool2d(1)
        s=pool(s)
        s=s.permute([2,1,0])
        s[...,0]=0.25+s[...,0]/10
        s[...,1]=0.25++s[...,1]/10
        s[...,2]=1-s[...,0]-s[...,1]
        if self.t%100==0:
            print(self.t,":",s)
        align=s[...,0]*align+s[...,1]*align2+s[...,2]*align3
        
        return torch.stack([rays_o, rays_d], dim=-1), torch.stack([rays_o2, rays_d2], dim=-1), torch.stack([rays_o3, rays_d3], dim=-1), weight, weight2, weight3, align, s

class NeRFAll(nn.Module):
    def __init__(self, args, kernelsnet=None):
        super().__init__()
        self.args = args
        self.embed_fn, self.input_ch = get_embedder(args.multires, args.i_embed)
        self.input_ch_views = 0
        self.kernelsnet = kernelsnet
        self.embeddirs_fn = None
        if args.use_viewdirs:
            self.embeddirs_fn, self.input_ch_views = get_embedder(args.multires_views, args.i_embed)

        self.output_ch = 5 if args.N_importance > 0 else 4

        skips = [4]
        self.mlp_coarse = NeRF(
            D=args.netdepth, W=args.netwidth,
            input_ch=self.input_ch, output_ch=self.output_ch, skips=skips,
            input_ch_views=self.input_ch_views, use_viewdirs=args.use_viewdirs)

        self.mlp_fine = None
        if args.N_importance > 0:
            self.mlp_fine = NeRF(
                D=args.netdepth_fine, W=args.netwidth_fine,
                input_ch=self.input_ch, output_ch=self.output_ch, skips=skips,
                input_ch_views=self.input_ch_views, use_viewdirs=args.use_viewdirs)

        activate = {'relu': torch.relu, 'sigmoid': torch.sigmoid, 'exp': torch.exp, 'none': lambda x: x,
                    'sigmoid1': lambda x: 1.002 / (torch.exp(-x) + 1) - 0.001,
                    'softplus': lambda x: nn.Softplus()(x - 1)}
        self.rgb_activate = activate[args.rgb_activate]
        self.sigma_activate = activate[args.sigma_activate]
        self.tonemapping = ToneMapping(args.tone_mapping_type)

    def mlpforward(self, inputs, viewdirs, mlp, netchunk=1024 * 64):
        """Prepares inputs and applies network 'fn'.
            """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = self.embed_fn(inputs_flat)

        if viewdirs is not None:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = self.embeddirs_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        # batchify execution
        if netchunk is None:
            outputs_flat = mlp(embedded)
        else:
            outputs_flat = torch.cat([mlp(embedded[i:i + netchunk]) for i in range(0, embedded.shape[0], netchunk)], 0)

        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs

    # 把模型的预测转化为有实际意义的表达，输入预测、时间和光束方向，输出光束颜色、视差、密度、每个采样点的权重和深度
    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """

        def raw2alpha(raw_, dists_, act_fn):
            alpha_ = - torch.exp(-act_fn(raw_) * dists_) + 1.
            return torch.cat([alpha_, torch.ones_like(alpha_[:, 0:1])], dim=-1)

        dists = z_vals[..., 1:] - z_vals[..., :-1]  # [N_rays, N_samples - 1]
        # dists = torch.cat([dists, torch.tensor([1e10]).expand(dists[..., :1].shape)], -1)

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = self.rgb_activate(raw[..., :3])
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn_like(raw[..., :-1, 3]) * raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
                noise = torch.tensor(noise)

        density = self.sigma_activate(raw[..., :-1, 3] + noise)
        if not self.training and self.args.render_rmnearplane > 0:
            mask = z_vals[:, 1:]
            mask = mask > self.args.render_rmnearplane / 128
            mask = mask.type_as(density)
            density = mask * density

        alpha = - torch.exp(- density * dists) + 1.
        alpha = torch.cat([alpha, torch.ones_like(alpha[:, 0:1])], dim=-1)

        # alpha = raw2alpha(raw[..., :-1, 3] + noise, dists, act_fn=self.sigma_activate)  # [N_rays, N_samples]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = alpha * \
                  torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), - alpha + (1. + 1e-10)], -1), -1)[:, :-1]

        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        depth_map = torch.sum(weights * z_vals, -1)

        # disp_map = 1. / torch.clamp_min(depth_map, 1e-10)
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map, density, acc_map, weights, depth_map

    def render_rays(self,
                    ray_batch,
                    N_samples,
                    retraw=False,
                    lindisp=False,
                    perturb=0.,
                    N_importance=0,
                    white_bkgd=False,
                    raw_noise_std=0.,
                    pytest=False):
        """Volumetric rendering.
        Args:
          ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
          N_samples: int. Number of different times to sample along each ray.
          retraw: bool. If True, include model's raw, unprocessed predictions.
          lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
          perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
          N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
          white_bkgd: bool. If True, assume a white background.
          raw_noise_std: ...
          verbose: bool. If True, print more debugging info.
        """
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

        t_vals = torch.linspace(0., 1., steps=N_samples).type_as(rays_o)
        if not lindisp:
            z_vals = near * (1. - t_vals) + far * (t_vals)
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).type_as(rays_o)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

        #     raw = run_network(pts)
        raw = self.mlpforward(pts, viewdirs, self.mlp_coarse)
        rgb_map, density_map, acc_map, weights, depth_map = self.raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                             white_bkgd, pytest=pytest)

        if N_importance > 0:
            rgb_map_0, depth_map_0, acc_map_0, density_map0 = rgb_map, depth_map, acc_map, density_map

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                                None]  # [N_rays, N_samples + N_importance, 3]

            mlp = self.mlp_coarse if self.mlp_fine is None else self.mlp_fine
            raw = self.mlpforward(pts, viewdirs, mlp)

            rgb_map, density_map, acc_map, weights, depth_map = self.raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                                 white_bkgd, pytest=pytest)

        ret = {'rgb_map': rgb_map, 'depth_map': depth_map, 'acc_map': acc_map, 'density_map': density_map}
        if retraw:
            ret['raw'] = raw
        if N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['depth0'] = depth_map_0
            ret['acc0'] = acc_map_0
            ret['density0'] = density_map0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        for k in ret:
            if torch.isnan(ret[k]).any():
                print(f"! [Numerical Error] {k} contains nan.")
            if torch.isinf(ret[k]).any():
                print(f"! [Numerical Error] {k} contains inf.")
        return ret

    def forward(self, H, W, K, chunk=1024 * 32, rays=None, rays_info=None, poses=None, **kwargs):
        """
        render rays or render poses, rays and poses should atleast specify one 渲染光线或渲染位姿，光线和位姿应至少指定一个
        calling model.train() to render rays, where rays, rays_info, should be specified 来渲染光线，其中应该指定光线rays_info
        calling model.eval() to render an image, where poses should be specified 来渲染图像，其中应指定位姿

        optional args:可选参数：
        force_naive: when True, will only run the naive NeRF, even if the kernelsnet is specified
                     当为True时，即使指定了内核，也只运行naive NeRF
        """
        # training
        if self.training:
            assert rays is not None, "Please specify rays when in the training mode"

            force_baseline = kwargs.pop("force_naive", True)
            # kernel mode, run multiple rays to get result of one ray
            # 内核模式，运行多条光线以获得一条光线的结果
            if self.kernelsnet is not None and not force_baseline:
                if self.kernelsnet.require_depth:
                    with torch.no_grad():
                        rgb, depth, acc, extras = self.render(H, W, K, chunk, rays, **kwargs)
                        rays_info["ray_depth"] = depth[:, None]

                # time0 = time.time()
                # 改改改改改改改
                new_rays, new_rays2, new_rays3, weight, weight2, weight3, align_loss, s  = self.kernelsnet(H, W, K, rays, rays_info)
               #  print(s)
                ray_num, pt_num = new_rays.shape[:2]
                ray_num2, pt_num2 = new_rays.shape[:2]
                ray_num3, pt_num3 = new_rays3.shape[:2]
                # time1 = time.time()
                rgb, depth, acc, extras = self.render(H, W, K, chunk, new_rays.reshape(-1, 3, 2), **kwargs)
                rgb2, depth2, acc2, extras2 = self.render(H, W, K, chunk, new_rays2.reshape(-1, 3, 2), **kwargs)
                rgb3, depth3, acc3, extras3 = self.render(H, W, K, chunk, new_rays3.reshape(-1, 3, 2), **kwargs)
                rgb_pts = rgb.reshape(ray_num, pt_num, 3)
                rgb_pts2 = rgb2.reshape(ray_num2, pt_num2, 3)
                rgb_pts3 = rgb3.reshape(ray_num3, pt_num3, 3)

                rgb0_pts = extras['rgb0'].reshape(ray_num, pt_num, 3)
                rgb0_pts2 = extras2['rgb0'].reshape(ray_num2, pt_num2, 3)
                rgb0_pts3 = extras3['rgb0'].reshape(ray_num3, pt_num3, 3)

                # time2 = time.time()
                rgb = torch.sum(rgb_pts * weight[..., None], dim=1)
                rgb0 = torch.sum(rgb0_pts * weight[..., None], dim=1)
                rgb = self.tonemapping(rgb)
                rgb0 = self.tonemapping(rgb0)

                rgb2 = torch.sum(rgb_pts2 * weight2[..., None], dim=1)
                rgb0_2 = torch.sum(rgb0_pts2 * weight2[..., None], dim=1)
                rgb2 = self.tonemapping(rgb2)
                rgb0_2 = self.tonemapping(rgb0_2)

                rgb3 = torch.sum(rgb_pts3 * weight3[..., None], dim=1)
                rgb0_3 = torch.sum(rgb0_pts3 * weight3[..., None], dim=1)
                rgb3 = self.tonemapping(rgb3)
                rgb0_3 = self.tonemapping(rgb0_3)
                # time3 = time.time()
                # print(f"Time| kernel: {time1-time0:.5f}, nerf: {time2-time1:.5f}, fuse: {time3-time2}")

               # rgb=s[0]*rgb+s[1]*rgb2+s[2]*rgb3
                #rgb0 =s[0] * rgb0 + s[1] * rgb0_2 + s[2] * rgb0_3

                rgb=s[...,0]*rgb+s[...,1]*rgb2+s[...,2]*rgb3

                rgb0 =s[...,0]*rgb0 +s[...,1]*rgb0_2 + s[...,2]*rgb0_3
                other_loss = {}

                # compute align loss, some priors of the ray pattern
                # ========================
                if align_loss is not None:
                    other_loss["align"] = align_loss.reshape(1, 1)
                

                return rgb, rgb0, other_loss
            # 改改改改改改改
            else:
                rgb, depth, acc, extras = self.render(H, W, K, chunk, rays, **kwargs)
                return self.tonemapping(rgb), self.tonemapping(extras['rgb0']), {}

        #  evaluation
        else:
            assert poses is not None, "Please specify poses when in the eval model"
            if "render_point" in kwargs.keys():
                rgbs, depths, weights = self.render_subpath(H, W, K, chunk, poses, **kwargs)
                depths = weights * 2
            else:
                rgbs, depths = self.render_path(H, W, K, chunk, poses, **kwargs)
            return self.tonemapping(rgbs), depths

    def render(self, H, W, K, chunk, rays=None, c2w=None, ndc=True,
               near=0., far=1.,
               use_viewdirs=False, c2w_staticcam=None,
               **kwargs):  # the render function
        """Render rays
            Args:
              H: int. Height of image in pixels.
              W: int. Width of image in pixels.
              focal: float. Focal length of pinhole camera.
              chunk: int. Maximum number of rays to process simultaneously. Used to
                control maximum memory usage. Does not affect final results.
              rays: array of shape [2, batch_size, 3]. Ray origin and direction for
                each example in batch.
              c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
              ndc: bool. If True, represent ray origin, direction in NDC coordinates.
              near: float or array of shape [batch_size]. Nearest distance for a ray.
              far: float or array of shape [batch_size]. Farthest distance for a ray.
              use_viewdirs: bool. If True, use viewing direction of a point in space in model.
              c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
               camera while using other c2w argument for viewing directions.将此变换矩阵用于相机，同时将其他c2w参数用于查看方向。
            Returns:
              rgb_map: [batch_size, 3]. Predicted RGB values for rays.
              disp_map: [batch_size]. Disparity map. Inverse of depth.
              acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
              extras: dict with everything returned by render_rays().
            """
        rays_o, rays_d = rays[..., 0], rays[..., 1]

        if use_viewdirs:
            # provide ray directions as input
            viewdirs = rays_d
            if c2w_staticcam is not None:
                # special case to visualize effect of viewdirs
                rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        sh = rays_d.shape  # [..., 3]
        if ndc:
            # for forward facing scenes
            rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)
        if use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)

        # Batchfy and Render and reshape
        all_ret = {}
        for i in range(0, rays.shape[0], chunk):
            ret = self.render_rays(rays[i:i + chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        k_extract = ['rgb_map', 'depth_map', 'acc_map']
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
        return ret_list + [ret_dict]

    def render_path(self, H, W, K, chunk, render_poses, render_kwargs, render_factor=0, ):
        """
        render image specified by the render_poses
        """
        if render_factor != 0:
            # Render downsampled for speed
            H = H // render_factor
            W = W // render_factor

        rgbs = []
        depths = []

        t = time.time()
        for i, c2w in enumerate(render_poses):
            print(i, time.time() - t)
            t = time.time()
            rays = get_rays(H, W, K, c2w)
            rays = torch.stack(rays, dim=-1)
            rgb, depth, acc, extras = self.render(H, W, K, chunk=chunk, rays=rays, c2w=c2w[:3, :4], **render_kwargs)

            rgbs.append(rgb)
            depths.append(depth)
            if i == 0:
                print(rgb.shape, depth.shape)

        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)

        return rgbs, depths

    def render_subpath(self, H, W, K, chunk, render_poses, render_point, images_indices, render_kwargs,
                       render_factor=0):
        """
        
        """
        if render_factor != 0:
            # Render downsampled for speed
            H = H // render_factor
            W = W // render_factor

        rgbs = []
        depths = []
        weights = []

        t = time.time()

        rayx, rayy = torch.meshgrid(torch.linspace(0, W - 1, W),
                                    torch.linspace(0, H - 1, H))
        rayx = rayx.t().reshape(-1, 1) + HALF_PIX
        rayy = rayy.t().reshape(-1, 1) + HALF_PIX

        for imgidx, c2w in zip(images_indices, render_poses):

            i = int(imgidx.item())
            print(i, time.time() - t)
            t = time.time()
            rays = get_rays(H, W, K, c2w)
            rays = torch.stack(rays, dim=-1).reshape(H * W, 3, 2)

            rays_info = {}

            if self.kernelsnet.require_depth:
                with torch.no_grad():
                    rgb, depth, acc, extras = self.render(H, W, K, chunk, rays, **render_kwargs)
                    rays_info["ray_depth"] = depth[..., None]

            i = i if i < self.kernelsnet.num_img else 1
            rays_info["images_idx"] = torch.ones_like(rays[:, 0:1, 0]).type(torch.long) * i
            rays_info["rays_x"] = rayx
            rays_info["rays_y"] = rayy

            new_rays, weight, _ = self.kernelsnet(H, W, K, rays, rays_info)

            new_rays = new_rays[:, render_point]
            weight = weight[:, render_point]
            rgb, depth, acc, extras = self.render(H, W, K, chunk=chunk, rays=new_rays.reshape(-1, 3, 2),
                                                  c2w=c2w[:3, :4], **render_kwargs)

            rgbs.append(rgb.reshape(H, W, 3))
            depths.append(depth.reshape(H, W))
            weights.append(weight.reshape(H, W))
            if i == 0:
                print(rgb.shape, depth.shape)

        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)
        weights = torch.stack(weights, 0)

        return rgbs, depths, weights
