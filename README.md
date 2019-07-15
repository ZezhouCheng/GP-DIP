## A Bayesian Perspective on the Deep Image Prior (CVPR 2019)

This repository contains the source code for the CVPR 2019 paper <u>A Bayesian Perspective on the Deep Image Prior</u>. 

[[Paper]](https://people.cs.umass.edu/~zezhoucheng/gp-dip/gp-dip.pdf)  [[Supplementary]](https://people.cs.umass.edu/~zezhoucheng/gp-dip/gp-dip-supp.pdf) [[arXiv]](https://arxiv.org/abs/1904.07457) [[Project page]](https://people.cs.umass.edu/~zezhoucheng/gp-dip/)  


### Installation

Our implementation is based on the code from Deep Image Prior [Ulyanov et al. CVPR 2018]. Refer to their [project page](https://github.com/DmitryUlyanov/deep-image-prior) for installation. (Dependencies: python = 3.6; pytorch = 0.4; numpy; 
scipy; matplotlib; scikit-image; jupyter; seaborn; \_pickle)

Recommanded way to install: 
```
conda create -n GP-DIP python=3.6 anaconda
source activate GP-DIP
cat /usr/local/cuda/version.txt # check out the CUDA version
conda install pytorch=0.4.1 cuda80 torchvision -c pytorch
```

To run the **GP_RBF_Inpainting.ipynb**, [gpytorch](https://github.com/cornellius-gp/gpytorch) is required (Dependencies: python >= 3.6; pyTorch >= 1.0).

### Tutorials

* **1D_toy_example.ipynb**: priors and posterior with 1D convolutional networks.

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
