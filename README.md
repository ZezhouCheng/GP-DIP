## A Bayesian Perspective on the Deep Image Prior (CVPR 2019)

This repository contains the source code for the CVPR 2019 paper <u>A Bayesian Perspective on the Deep Image Prior</u>. 

[[Paper]]()  [[Supplementary]]() [[arXiv]]() [[Project page]](https://people.cs.umass.edu/~zezhoucheng/gp-dip/)  


### Installation

Our implementation is based on the code from Deep Image Prior [Ulyanov et al. CVPR 2018]. Refer to their [project page](https://github.com/DmitryUlyanov/deep-image-prior) for installation. 

To run the **GP_RBF_Inpainting.ipynb**, [gpytorch](https://github.com/cornellius-gp/gpytorch) is required.

### Notebook tutorials

* **1D_toy_example.ipynb**: compare the analytical and numerical kernel; CNN prior and SGLD posterior

* **Image_Denoising.ipynb**: compare the SGD variants and SGLD on image denoising task

* **Image_Inpainting.ipynb**: compare the SGD variants and SGLD on image inpainting task


* **GP_RBF_Inpainting.ipynb**: the Gaussian Process with RBF kernel for image inpainting.
* **GP_DIP_Inpainting.ipynb**: the Gaussian Process with DIP kernel for image inpainting.


### Datasets 

Download the dataset [here](https://www.dropbox.com/sh/etej8iipw4fa75g/AABAA84Ng-ZqmJHNAVN6Bi5pa?dl=0) for the large-hole image inpainting experiments presented in our supplementary.

### Citation

```
@inproceedings{Cheng_2019_CVPR,
	author = {Cheng, Zezhou and Gadelha, Matheus and Maji, Subhransu and Sheldon, Daniel},
	title = {A Bayesian Perspective on the Deep Image Prior},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	year = {2019}
}
```
