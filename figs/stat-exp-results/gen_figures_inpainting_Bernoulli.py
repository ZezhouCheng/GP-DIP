import numpy as np
import _pickle as cPickle
from PIL import Image
import PIL
import os

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



imlist = ['barbara', 'boat','house','Lena512','peppers256','Cameraman256', 'couple','fingerprint',     'hill',  'man', 'montage']

for run_id in range(10): # 10: run 100 times

    print(run_id)
    path = '%d/' % run_id
    for imname in imlist:
        
        dirname = path + imname
        output_dir = mkdir_if_not_exist(dirname+'/output')
        
        with open(dirname + '/mask_and_image', 'rb') as f:
            img_np = cPickle.load(f)
            masked_img_np = cPickle.load(f)
            img_mask_np = cPickle.load(f)


        ## 'XXXX_2' means blending the known parts with the predictions fom CNN on the masked parts
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
            
        img = np_to_pil(img_np)
        masked_img = np_to_pil(masked_img_np)

        pil_sgd_out_5000 = np_to_pil(sgd_out_5000)
        pil_sgd_out_5000_2 = np_to_pil(sgd_out_5000_2)
        pil_sgd_expm_out_5000 = np_to_pil(sgd_expm_out_5000)
        pil_sgd_expm_out_5000_2 = np_to_pil(sgd_expm_out_5000_2)

        pil_sgd_out_11000 = np_to_pil(fancy_sgd_out_11000)
        pil_sgd_out_11000_2 = np_to_pil(fancy_sgd_out_11000_2)
        pil_sgd_expm_out_11000 = np_to_pil(fancy_sgd_expm_out_11000)
        pil_sgd_expm_out_11000_2 = np_to_pil(fancy_sgd_expm_out_11000_2)

        pil_sgld_mean = np_to_pil(sgld_mean)
        pil_sgld_mean_2 = np_to_pil(sgld_mean_2)

        # save images 
        img.save(output_dir + '/gt.png')
        masked_img.save(output_dir + '/masked_image.png')

        pil_sgd_out_5000.save(output_dir + '/sgd_5000.png')
        pil_sgd_out_5000_2.save(output_dir + '/sgd_5000_2.png')
        pil_sgd_expm_out_5000.save(output_dir + '/sgd_expm_5000.png')
        pil_sgd_expm_out_5000_2.save(output_dir + '/sgd_expm_5000_2.png')

        pil_sgd_out_11000.save(output_dir + '/fancy_sgd_11000.png')
        pil_sgd_out_11000_2.save(output_dir + '/fancy_sgd_11000_2.png') 
        pil_sgd_expm_out_11000.save(output_dir + '/fancy_sgd_expm_11000.png') 
        pil_sgd_expm_out_11000_2.save(output_dir + '/fancy_sgd_expm_11000_2.png') 

        pil_sgld_mean.save(output_dir + '/sgld_mean.png') 
        pil_sgld_mean_2.save(output_dir + '/sgld_mean_2.png') 
       

        # print PSNR
        check_point = 5000 - 1
        f = open(output_dir + '/PSNR.txt', 'w')


        f.write('sgd_5000: %.2f\n' % sgd_psnr_list[check_point])
        f.write('sgd_expm_5000: %.2f\n' % sgd_expm_psnr_list[check_point])
        f.write('sgd_5000_2: %.2f\n' % sgd_psnr_list2[check_point])
        f.write('sgd_expm_5000_2: %.2f\n' % sgd_expm_psnr_list2[check_point])


        check_point = 11000 - 1
        f.write('fancy_sgd_11000: %.2f\n' % fancy_sgd_psnr_list[check_point])
        f.write('fancy_sgd_expm_11000: %.2f\n' % fancy_sgd_expm_psnr_list[check_point])
        f.write('fancy_sgd_11000_2: %.2f\n' % fancy_sgd_psnr_list2[check_point])
        f.write('fancy_sgd_expm_11000_2: %.2f\n' % fancy_sgd_expm_psnr_list2[check_point])
       

        f.write('sgld_mean_psnr: %.2f\n' % sgld_mean_psnr)
        f.write('sgld_mean_psnr_2: %.2f\n' % sgld_mean_psnr_2)

