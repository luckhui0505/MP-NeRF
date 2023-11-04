# MP-NeRF
MP-NeRF takes a unique approach by incorporating distinct priors tailored to different types of blur, enhancing its ability to understand the blur formation process. Our method significantly improves PSNR, SSIM and LPIPS metrics. Especially, the improvement of LPIPS is the most obvious, we can see that LPIPS improves about 32.1\% on average compared to Deblur-NeRF in camera motion blur, and about 24.8\% on average compared to Deblur-NeRF in defocus blur.

## Comparison of Experimental Results
![image](https://github.com/luckhui0505/MP-NeRF/blob/master/result.png) 
Quantitative results on synthetic scenes of two blur types. We bolded the best result in the metric.
## Method Overview
![image](https://github.com/luckhui0505/MP-NeRF/blob/master/framework.png) 
The overall network structure of MP-NeRF. When rendering a ray , it first input to S branches, and each branch predicts the offsets and their weights for N sparse optimisation rays in combination with view embedding. Each branch renders the ray through NeRF and the result can be obtained. PLW network provides weight coefficients for each branch.

## Quick Start
### 1.Install environment
' ''
cd MP-NeRF
pip install -r requirements.txt
' ''
### 2. Download dataset
There are total of 31 scenes used in the paper. We mainly focus on camera motion blur and defocus blur, so we use 5 synthetic scenes and 10 real world scenes for each blur type. We also include one case of object motion blur. You can download all the data in [here](https://hkustconnect-my.sharepoint.com/personal/lmaag_connect_ust_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Flmaag%5Fconnect%5Fust%5Fhk%2FDocuments%2Fshare%2FCVPR2022%2Fdeblurnerf%5Fdataset&ga=1).
### 3. Setting parameters
Changing the data path and log path in the configs/demo_blurball.txt

