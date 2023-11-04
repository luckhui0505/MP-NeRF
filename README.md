# MP-NeRF
MP-NeRF takes a unique approach by incorporating distinct priors tailored to different types of blur, enhancing its ability to understand the blur formation process. Our method significantly improves PSNR, SSIM and LPIPS metrics. Especially, the improvement of LPIPS is the most obvious, we can see that LPIPS improves about 32.1\% on average compared to Deblur-NeRF in camera motion blur, and about 24.8\% on average compared to Deblur-NeRF in defocus blur.

## Comparison of Experimental Results
![image](https://github.com/luckhui0505/MP-NeRF/blob/master/result.png) 
Quantitative results on synthetic scenes of two blur types. We bolded the best result in the metric.
## Method Overview
![image](https://github.com/luckhui0505/MP-NeRF/blob/master/framework.png) 
The overall network structure of MP-NeRF. When rendering a ray , it first input to S branches, and each branch predicts the offsets and their weights for N sparse optimisation rays in combination with view embedding. Each branch renders the ray through NeRF and the result can be obtained. PLW network provides weight coefficients for each branch.

## Quick Start

