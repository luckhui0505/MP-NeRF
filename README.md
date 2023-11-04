# MP-NeRF
MP-NeRF takes a unique approach by incorporating distinct priors tailored to different types of blur, enhancing its ability to understand the blur formation process. Additionally, MP-NeRF introduces a Multi-branch Fusion Network (MBFNet) in combination with a Prior-based Learnable Network (PLW). These components work together to capture the intricate details of blurry images, including geometric features like textures and patterns. 

Our method significantly improves PSNR, SSIM and LPIPS metrics. Especially, the improvement of LPIPS is the most obvious, we can see that LPIPS improves about 32.1\% on average compared to Deblur-NeRF in camera motion blur, and about 24.8\% on average compared to Deblur-NeRF in defocus blur.
