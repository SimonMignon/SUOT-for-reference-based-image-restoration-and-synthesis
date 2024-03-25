# Semi-Unbalanced Optimal Transport for Reference-Based Image Restoration and Synthesis

## Overview
This repository contains the code necessary to replicate most of the figures featured in our article [1]. A preprint of the article is accessible at [HAL-04514983](https://hal.science/hal-04514983). If our code or methodology aids in your research, kindly cite our paper.

Portions of the codebase were adapted from existing sources [2],[3] and [4] which have been modified to fit our semi-unbalanced formulation of optimal transport. We extend our gratitude to the original authors.

## Requirements
To run the notebooks in this repository, you'll need Python along with several packages. While the code is likely compatible with other versions of these packages, it was developed and tested with the following principal configurations:

- **WPP (gray images and color images), WPPNets (color images), PSinOT:**
  - PyTorch: 1.9.0+cu111
  - PyKeOps: 2.1.2

- **WPPNets (gray images):**
  - PyTorch: 1.10.0
  - PyKeOps: 1.5

## References 
- [1] Simon Mignon, Bruno Galerne, Moncef Hidane, Cécile Louchet, Julien Mille. Semi-Unbalanced Optimal Transport for Image Restoration and Synthesis. 2024.
- [2] J. Hertrich, A. Houdard, and C. Redenbach. Wasserstein patch prior for image superresolution. IEEE Transactions on Computational Imaging, 8:693–704, 2022.
- [3] Fabian Altekrüger and Johannes Hertrich. WPPNets and WPPFlows: The power of Wasserstein patch priors for superresolution. SIAM Journal on Imaging Sciences, 16(3):1033–1067, 2023.
- [4] Nicolas Cherel, Andrés Almansa, Yann Gousseau, and Alasdair Newson. A patch-based algorithm for diverse and high fidelity single image generation. In 29th IEEE International Conference on Image Processing (ICIP) 2022, Bordeaux, France, October 2022.


