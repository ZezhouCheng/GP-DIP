import matplotlib
matplotlib.use('Agg')
import time
import  matplotlib.pyplot as plt
import numpy as np
import _pickle as cPickle
import seaborn as sns


sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    
## load image

dirname = 'TODO' ## indicate a dir that contains results
plt.rcParams.update({'font.size': 20})

with open(dirname + '/vanilla_SGD_data', 'rb') as f:
    sgd_noise_psnr_list = cPickle.load(f)
    sgd_psnr_list = cPickle.load(f)
    sgd_expm_psnr_list = cPickle.load(f)

with open(dirname + '/vanilla_SGD_data_weight_decay', 'rb') as f:
    sgd_noise_psnr_list2 = cPickle.load(f)
    sgd_psnr_list2 = cPickle.load(f)
    sgd_expm_psnr_list2 = cPickle.load(f)


with open(dirname +'/fancy_SGD_data', 'rb') as f:
    fancy_sgd_noise_psnr_list = cPickle.load(f)
    fancy_sgd_psnr_list = cPickle.load(f)
    fancy_sgd_expm_psnr_list = cPickle.load(f)

with open(dirname +'/Gaussian_SGLD_data', 'rb') as f:
    sgld_psnr_list = cPickle.load(f)
    sgld_psnr_sm_list = cPickle.load(f)
    sgld_mean_psnr = cPickle.load(f)
    sgld_mean = cPickle.load(f)
    sgld_psnr_mean_list = cPickle.load(f)

plt.figure(1, figsize=(9,5))    
num_iter = len(sgd_psnr_list)
num_iter = 20000

sgd_psnr_list = np.array(sgd_psnr_list)
sgd_expm_psnr_list = np.array(sgd_expm_psnr_list)
sgd_psnr_list2 = np.array(sgd_psnr_list2)
fancy_sgd_psnr_list = np.array(fancy_sgd_psnr_list)
fancy_sgd_expm_psnr_list = np.array(fancy_sgd_expm_psnr_list)
sgld_psnr_mean_list = np.array(sgld_psnr_mean_list)
sgld_psnr_list = np.array(sgld_psnr_list)


x_iters = np.arange(0, num_iter, 50)
plt.plot(x_iters, sgd_psnr_list[x_iters], color = (1, 0, 0), label = 'SGD')
plt.plot(x_iters, sgd_expm_psnr_list[x_iters], 'r--', label = 'SGD+Avg')

# with weight decay
plt.plot(x_iters, sgd_psnr_list2[x_iters], color = 'k', label = 'SGD+WD')

plt.plot(x_iters, fancy_sgd_psnr_list[x_iters], 'y', label = 'SGD+Input')
plt.plot(x_iters, fancy_sgd_expm_psnr_list[x_iters],'y--', label = 'SGD+Input+Avg')
burnin_iter = 7000
x_iters_after_burnin = np.arange(len(sgld_psnr_mean_list[:num_iter])) + burnin_iter 

plt.plot(x_iters, sgld_psnr_list[x_iters], 'b', label = 'SGLD')
plt.plot(np.arange(7000, num_iter-1), sgld_psnr_mean_list, 'b--', label = 'SGLD Avg w.r.t. Iters')
plt.plot([0, 5000, 10000, 15000, 20000], [sgld_mean_psnr] * 5, '--bo', label = 'SGLD Avg')

plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0001), ncol = 4, prop={'size':10.5})

timestr = time.strftime("%m%d-%H%M%S")
plt.xlabel('Iteration')
plt.xticks([0, 5000, 10000, 15000, 20000], ('0', '5K', '10K', '15K', '20K'))
plt.ylim(20, 31)
plt.ylabel('PSNR (dB)')
plt.savefig('PSNR.pdf', bbox_inches='tight')
plt.show()
