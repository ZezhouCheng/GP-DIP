import numpy as np
import _pickle as cPickle
from PIL import Image
import PIL
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def mkdir_if_not_exist(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok = True)
    return dirname


def np_to_pil(img_np): 
    '''Converts image in np.array format to PIL image.
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)
    return Image.fromarray(ar)


root = '../data/inpainting/Dataset_large_hole/'
img_dir = root + 'image/'
mask_dir = root + 'mask/'
imlist = os.listdir(img_dir)

root_dir = 'TODO' ## indicate the path that contains the experimental results

# generate inpainting results
for imname in imlist:
    for run_id in range(2):
        for method in ['sgd', 'sgld']:
        
            ffname = imname[:-4]
            print(ffname, method)
            dirname = root_dir + ffname + '_' + 'fancy_sgd' + '_%d' % run_id

            print(dirname)
            output_dir = mkdir_if_not_exist('output_fancy2/'+ffname)
            
            # with open(dirname + '/mask_and_image', 'rb') as f:
            #    img_np = cPickle.load(f)
            #    masked_img_np = cPickle.load(f)
            #    img_mask_np = cPickle.load(f)

            if method == 'sgd':
            
                vanilla_sgd_data = dirname + '/vanilla_SGD_data'
                fancy_sgd_data = dirname + '/fancy_SGD_data'

                if not os.path.exists(vanilla_sgd_data) or not os.path.exists(fancy_sgd_data):
                    continue
            
                with open(vanilla_sgd_data, 'rb') as f:
                    sgd_psnr_list=cPickle.load(f)
                    sgd_expm_psnr_list=cPickle.load( f)
                    sgd_psnr_list2=cPickle.load(f)
                    sgd_expm_psnr_list2=cPickle.load(f)
                    sgd_out_5000=cPickle.load(f)
                    sgd_expm_out_5000=cPickle.load(f)
                    sgd_out_5000_2=cPickle.load(f) 
                    sgd_expm_out_5000_2=cPickle.load(f) 


                with open(fancy_sgd_data, 'rb') as f:
                    fancy_sgd_psnr_list=cPickle.load(f)
                    fancy_sgd_expm_psnr_list=cPickle.load(f)
                    fancy_sgd_psnr_list2=cPickle.load(f)
                    fancy_sgd_expm_psnr_list2=cPickle.load(f)
                    fancy_sgd_out_11000=cPickle.load(f)
                    fancy_sgd_expm_out_11000=cPickle.load(f)
                    fancy_sgd_out_11000_2 =  cPickle.load(f)
                    fancy_sgd_expm_out_11000_2 = cPickle.load(f)

                pil_sgd_out_5000 = np_to_pil(sgd_out_5000)
                pil_sgd_out_5000_2 = np_to_pil(sgd_out_5000_2)
                pil_sgd_expm_out_5000 = np_to_pil(sgd_expm_out_5000)
                pil_sgd_expm_out_5000_2 = np_to_pil(sgd_expm_out_5000_2)

                pil_sgd_out_11000 = np_to_pil(fancy_sgd_out_11000)
                pil_sgd_out_11000_2 = np_to_pil(fancy_sgd_out_11000_2)
                pil_sgd_expm_out_11000 = np_to_pil(fancy_sgd_expm_out_11000)
                pil_sgd_expm_out_11000_2 = np_to_pil(fancy_sgd_expm_out_11000_2)

                pil_sgd_out_5000.save(output_dir + '/sgd_5000.png')
                # pil_sgd_out_5000_2.save(output_dir + '/sgd_5000_2.png')
                pil_sgd_expm_out_5000.save(output_dir + '/sgd_expm_5000.png')
                # pil_sgd_expm_out_5000_2.save(output_dir + '/sgd_expm_5000_2.png')

                pil_sgd_out_11000.save(output_dir + '/fancy_sgd_11000.png')
                # pil_sgd_out_11000_2.save(output_dir + '/fancy_sgd_11000_2.png') 
                pil_sgd_expm_out_11000.save(output_dir + '/fancy_sgd_expm_11000.png') 
                # pil_sgd_expm_out_11000_2.save(output_dir + '/fancy_sgd_expm_11000_2.png') 



                check_point = 2001 - 1
                f = open(output_dir + '/PSNR_SGD.txt', 'w')
                f.write('sgd_5000: %.2f\n' % sgd_psnr_list[check_point])
                f.write('sgd_expm_5000: %.2f\n' % sgd_expm_psnr_list[check_point])
                f.write('sgd_5000_2: %.2f\n' % sgd_psnr_list2[check_point])
                f.write('sgd_expm_5000_2: %.2f\n' % sgd_expm_psnr_list2[check_point])

                check_point = 3001 - 1
                f.write('fancy_sgd_11000: %.2f\n' % fancy_sgd_psnr_list[check_point])
                f.write('fancy_sgd_expm_11000: %.2f\n' % fancy_sgd_expm_psnr_list[check_point])
                f.write('fancy_sgd_11000_2: %.2f\n' % fancy_sgd_psnr_list2[check_point])
                f.write('fancy_sgd_expm_11000_2: %.2f\n' % fancy_sgd_expm_psnr_list2[check_point])

            else:

                gaussian_sgld_data = dirname + '/Gaussian_SGLD_data'
                if not os.path.exists(gaussian_sgld_data):
                    continue
                with open(gaussian_sgld_data, 'rb') as f: 
                    sgld_psnr_list = cPickle.load(f)
                    sgld_psnr_list2=cPickle.load(f)
                    sgld_mean=cPickle.load(f)
                    sgld_mean_2=cPickle.load(f)
                    sgld_mean_psnr=cPickle.load(f)
                    sgld_mean_psnr_2=cPickle.load(f)

                pil_sgld_mean = np_to_pil(sgld_mean)
                pil_sgld_mean_2 = np_to_pil(sgld_mean_2)

                pil_sgld_mean.save(output_dir + '/sgld_mean.png') 
                pil_sgld_mean_2.save(output_dir + '/sgld_mean_2.png') 

                f = open(output_dir + '/PSNR_SGLD.txt', 'w')
                f.write('sgld_mean_psnr: %.2f\n' % sgld_mean_psnr)
                f.write('sgld_mean_psnr_2: %.2f\n' % sgld_mean_psnr_2)

                # generate uncertainty map
                dir_samples = dirname + '/samples/'
                samples = []
                sample_psnr_list = []
                xiters = np.arange(20050, 30000, 50)
                for i in xiters:
                    file = dir_samples + '%d' % i
                    with open(file, 'rb') as f:
                        sample = cPickle.load(f)
                        samples.append(sample)
                uncertainty_map = np.mean(np.std(np.array(samples), axis=0), axis = 0)
                plt.imsave(output_dir + '/uncertainty_map.png', uncertainty_map, cmap=cm.RdYlGn)





