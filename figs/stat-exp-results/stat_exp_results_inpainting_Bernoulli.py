import _pickle as cPickle
import numpy as np
np.set_printoptions(precision=2)
import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pylab as plt

imlist = ['barbara', 'boat','house','Lena512','peppers256','Cameraman256', 'couple','fingerprint',     'hill',  'man', 'montage']

root = 'TODO' ## indicate the path the contains the experimental results

sgd_5000_list = []
sgd_expm_5000_list = []
sgd_5000_list2 = []
sgd_expm_5000_list2 = []


fancy_sgd_11000_list = []
fancy_sgd_expm_11000_list = []
fancy_sgd_11000_list2 = []
fancy_sgd_expm_11000_list2 = []


sgld_mean_list = []
sgld_mean_list2 = []
 
for run_id in range(10):
    path = root + '%d/' % run_id

    each_sgd_5000_list = []
    each_sgd_expm_5000_list = []
    each_sgd_5000_list2 = []
    each_sgd_expm_5000_list2 = []


    each_fancy_sgd_11000_list = []
    each_fancy_sgd_expm_11000_list = []
    each_fancy_sgd_11000_list2 = []
    each_fancy_sgd_expm_11000_list2 = []


    each_sgld_mean_list = []
    each_sgld_mean_list2 = []
    

    for imname in imlist:
        dirname = path + imname

        with open(dirname + '/vanilla_SGD_data', 'rb') as f:
            sgd_psnr_list=cPickle.load(f)
            sgd_expm_psnr_list=cPickle.load( f)
            sgd_psnr_list2=cPickle.load(f)
            sgd_expm_psnr_list2=cPickle.load(f)
            sgd_out_5000=cPickle.load(f)
            sgd_expm_out_5000=cPickle.load(f)
            sgd_out_5000_2=cPickle.load(f) 
            sgd_expm_out_5000_2=cPickle.load(f) 


        with open(dirname + '/fancy_SGD_data', 'rb') as f:
            fancy_sgd_psnr_list=cPickle.load(f)
            fancy_sgd_expm_psnr_list=cPickle.load(f)
            fancy_sgd_psnr_list2=cPickle.load(f)
            fancy_sgd_expm_psnr_list2=cPickle.load(f)
            fancy_sgd_out_11000=cPickle.load(f)
            fancy_sgd_expm_out_11000=cPickle.load(f)
            fancy_sgd_out_11000_2 =  cPickle.load(f)
            fancy_sgd_expm_out_11000_2 = cPickle.load(f)


        with open(dirname + '/Gaussian_SGLD_data', 'rb') as f: 
            sgld_psnr_list = cPickle.load(f)
            sgld_psnr_list2=cPickle.load(f)
            sgld_mean=cPickle.load(f)
            sgld_mean_2=cPickle.load(f)
            sgld_mean_psnr=cPickle.load(f)
            sgld_mean_psnr_2=cPickle.load(f)
            

        each_sgd_5000_list.append(sgd_psnr_list[5000 - 1])
        each_sgd_expm_5000_list.append(sgd_expm_psnr_list[5000 - 1])
        each_sgd_5000_list2.append(sgd_psnr_list2[5000 - 1])
        each_sgd_expm_5000_list2.append(sgd_expm_psnr_list2[5000 - 1])

        each_fancy_sgd_11000_list.append(fancy_sgd_psnr_list[11000 - 1])
        each_fancy_sgd_expm_11000_list.append(fancy_sgd_expm_psnr_list[11000 - 1])
        each_fancy_sgd_11000_list2.append(fancy_sgd_psnr_list2[11000 - 1])
        each_fancy_sgd_expm_11000_list2.append(fancy_sgd_expm_psnr_list2[11000 - 1])


        each_sgld_mean_list.append(sgld_mean_psnr)
        each_sgld_mean_list2.append(sgld_mean_psnr_2)



    sgd_5000_list.append(each_sgd_5000_list)
    sgd_expm_5000_list.append(each_sgd_expm_5000_list)
    sgd_5000_list2.append(each_sgd_5000_list2)
    sgd_expm_5000_list2.append(each_sgd_expm_5000_list2)

    fancy_sgd_11000_list.append(each_fancy_sgd_11000_list)
    fancy_sgd_expm_11000_list.append(each_fancy_sgd_expm_11000_list)
    fancy_sgd_11000_list2.append(each_fancy_sgd_11000_list2)
    fancy_sgd_expm_11000_list2.append(each_fancy_sgd_expm_11000_list2)


    sgld_mean_list.append(each_sgld_mean_list)
    sgld_mean_list2.append(each_sgld_mean_list2)



print('********************************')
print('\n')

print('~~~~~~~~wo known mean~~~~~~~~~~')
print(np.mean(sgd_5000_list, axis = 0))
print(np.mean(sgd_expm_5000_list, axis = 0))
print(np.mean(fancy_sgd_11000_list, axis = 0))
print(np.mean(fancy_sgd_expm_11000_list, axis = 0))
print(np.mean(sgld_mean_list, axis = 0))


print('~~~~~~~~wo known std~~~~~~~~~~')
print(np.std(sgd_5000_list, axis = 0))
print(np.std(sgd_expm_5000_list, axis = 0))
print(np.std(fancy_sgd_11000_list, axis = 0))
print(np.std(fancy_sgd_expm_11000_list, axis = 0))
print(np.std(sgld_mean_list, axis = 0))

print('********************************')
print('\n')

print('~~~~~~~~~with known~~~~~~~~~')
print(np.mean(sgd_5000_list2, axis = 0))
print(np.mean(sgd_expm_5000_list2, axis = 0))
print(np.mean(fancy_sgd_11000_list2, axis = 0))
print(np.mean(fancy_sgd_expm_11000_list2, axis = 0))
print(np.mean(sgld_mean_list2, axis = 0))

print('~~~~~~~~with known std~~~~~~~~~~')
print(np.std(sgd_5000_list2, axis = 0))
print(np.std(sgd_expm_5000_list2, axis = 0))
print(np.std(fancy_sgd_11000_list2, axis = 0))
print(np.std(fancy_sgd_expm_11000_list2, axis = 0))
print(np.std(sgld_mean_list2, axis = 0))

print('********************************')
print('\n')

print('~~~~~~~wo known mean of each method~~~~~~~~~~')
print(np.mean(sgd_5000_list))
print(np.mean(sgd_expm_5000_list))
print(np.mean(fancy_sgd_11000_list))
print(np.mean(fancy_sgd_expm_11000_list))
print(np.mean(sgld_mean_list))

print('~~~~~~~~~wo known std of each method~~~~~~~~~~')
print(np.std(np.mean(sgd_5000_list, axis = 1)))
print(np.std(np.mean(sgd_expm_5000_list, axis = 1)))
print(np.std(np.mean(fancy_sgd_11000_list, axis = 1)))
print(np.std(np.mean(fancy_sgd_expm_11000_list, axis = 1)))
print(np.std(np.mean(sgld_mean_list, axis = 1)))


print('********************************')
print('\n')

print('~~~~~~~with known mean of each method~~~~~~~~~~')
print(np.mean(sgd_5000_list2))
print(np.mean(sgd_expm_5000_list2))
print(np.mean(fancy_sgd_11000_list2))
print(np.mean(fancy_sgd_expm_11000_list2))
print(np.mean(sgld_mean_list2))

print('~~~~~~~~~with known std of each method~~~~~~~~~~')
print(np.std(np.mean(sgd_5000_list2, axis = 1)))
print(np.std(np.mean(sgd_expm_5000_list2, axis = 1)))
print(np.std(np.mean(fancy_sgd_11000_list2, axis = 1)))
print(np.std(np.mean(fancy_sgd_expm_11000_list2, axis = 1)))
print(np.std(np.mean(sgld_mean_list2, axis = 1)))


