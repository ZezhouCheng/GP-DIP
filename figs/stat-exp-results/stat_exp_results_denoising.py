
import _pickle as cPickle
import numpy as np
np.set_printoptions(precision=2)

import matplotlib
matplotlib.use('Agg')
import os

import matplotlib.pylab as plt

imlist = [ 'image_House256rgb','image_Peppers512rgb', 'image_Lena512rgb', 'image_Baboon512rgb', 'image_F16_512rgb', 'kodim01','kodim02', 'kodim03', 'kodim12']

sgld_avg = 0
sgld_no_std_avg = 0
sgld_avg2 = 0

root = '' ## indicate the path the contains the experimental results

sgd_500_list = []
sgd_expm_500_list = []
fancy_sgd_1800_list = []
fancy_sgd_expm_1800_list = []
sgld_mean_list = []

for run_id in range(10):

    path = root + '%d/' % run_id
    each_sgd_500_list = []
    each_sgd_expm_500_list = []
    each_fancy_sgd_1800_list = []
    each_fancy_sgd_expm_1800_list = []
    each_sgld_mean_list = []

    for imname in imlist:
        dirname = path + imname

        
        with open(dirname + '/noise_and_image', 'rb') as f:
            img_np = cPickle.load(f)
            img_noisy_np = cPickle.load(f)


        with open(dirname + '/vanilla_SGD_data', 'rb') as f:
            sgd_noisy_psnr_list = cPickle.load(f)
            sgd_psnr_list=cPickle.load(f)
            sgd_expm_psnr_list=cPickle.load( f)
            sgd_out_500=cPickle.load(f)
            sgd_expm_out_500=cPickle.load(f)


        with open(dirname + '/fancy_SGD_data', 'rb') as f:
            fancy_sgd_noisy_psnr_list = cPickle.load(f)
            fancy_sgd_psnr_list=cPickle.load(f)
            fancy_sgd_expm_psnr_list=cPickle.load( f)
            fancy_sgd_out_1800=cPickle.load(f)
            fancy_sgd_expm_out_1800=cPickle.load(f)


        with open(dirname + '/Gaussian_SGLD_data', 'rb') as f: 
            sgld_psnr_list = cPickle.load(f)
            sgld_psnr_sm_list = cPickle.load(f)
            sgld_mean_psnr=cPickle.load(f)
            sgld_mean=cPickle.load(f)
            sgld_psnr_mean_list=cPickle.load(f)
            


        each_sgd_500_list.append(sgd_psnr_list[499])
        each_sgd_expm_500_list.append(sgd_expm_psnr_list[499])
        each_fancy_sgd_1800_list.append(fancy_sgd_psnr_list[1799])
        each_fancy_sgd_expm_1800_list.append(fancy_sgd_expm_psnr_list[1799])
        each_sgld_mean_list.append(sgld_mean_psnr)

    sgd_500_list.append(each_sgd_500_list)
    sgd_expm_500_list.append(each_sgd_expm_500_list)
    fancy_sgd_1800_list.append(each_fancy_sgd_1800_list)
    fancy_sgd_expm_1800_list.append(each_fancy_sgd_expm_1800_list)
    sgld_mean_list.append(each_sgld_mean_list)


print('~~~~~~~~~~~~~~~~~~~')
print(np.mean(sgd_500_list, axis = 0))
print(np.mean(sgd_expm_500_list, axis = 0))
print(np.mean(fancy_sgd_1800_list, axis = 0))
print(np.mean(fancy_sgd_expm_1800_list, axis = 0))
print(np.mean(sgld_mean_list, axis = 0))

print('~~~~~~~~~~~~~~~~~~~')
print(np.std(sgd_500_list, axis = 0))
print(np.std(sgd_expm_500_list, axis = 0))
print(np.std(fancy_sgd_1800_list, axis = 0))
print(np.std(fancy_sgd_expm_1800_list, axis = 0))
print(np.std(sgld_mean_list, axis = 0))

print('~~~~~~~~~~~~~~~~~~~')
print(np.mean(sgd_500_list, axis = 1))
print(np.mean(sgd_expm_500_list, axis = 1))
print(np.mean(fancy_sgd_1800_list, axis = 1))
print(np.mean(fancy_sgd_expm_1800_list, axis = 1))
print(np.mean(sgld_mean_list, axis = 1))


print('~~~~~~~~~~~~~~~~~~~')
print(np.mean(sgd_500_list))
print(np.mean(sgd_expm_500_list))
print(np.mean(fancy_sgd_1800_list))
print(np.mean(fancy_sgd_expm_1800_list))
print(np.mean(sgld_mean_list))

print('~~~~~~~~~~~~~~~~~~~')
print(np.std(np.mean(sgd_500_list, axis = 1)))
print(np.std(np.mean(sgd_expm_500_list, axis = 1)))
print(np.std(np.mean(fancy_sgd_1800_list, axis = 1)))
print(np.std(np.mean(fancy_sgd_expm_1800_list, axis = 1)))
print(np.std(np.mean(sgld_mean_list, axis = 1)))


print(np.max(sgld_mean_list, axis = 0))
print(np.mean(np.max(sgld_mean_list, axis = 0)))















