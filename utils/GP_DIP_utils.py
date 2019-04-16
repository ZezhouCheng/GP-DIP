## import libs
from __future__ import print_function
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import numpy as np
from skimage.measure import compare_psnr
from utils.inpainting_utils import * 
from models import *
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

## display images
def np_plot(np_matrix, title, opt = 'RGB', savepath = None):
    plt.figure(2)
    plt.clf()
    if opt == 'RGB':
        fig = plt.imshow(np_matrix.transpose(1, 2, 0), interpolation = 'nearest')
    elif opt == 'map':
        fig = plt.imshow(np_matrix, interpolation = 'bilinear', cmap = cm.RdYlGn)
    elif opt == 'Grayscale':
        plt.gray()
        fig = plt.imshow(np.squeeze(np_matrix.transpose(1, 2, 0)),cmap=plt.get_cmap('gray'))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title(title)
    plt.axis('off')
    plt.pause(0.05)  
    if savepath is not None:
        plt.savefig(os.path.join(savepath, title+'.png'))
    
    
def load_test_data_inpainting():
    data = np.load("data/inpainting/test_data_inpainting.npz")
    img_np = data['img_np']
    img_mask_np = data['img_mask_np']
    train_x = data['train_x']
    train_y = data['train_y']
    test_x = data['test_x']

    return img_np, img_mask_np, train_x, train_y, test_x
    

# sampling
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def prior_sampling(num_samples = 2000, map_size = 500):
    
    input_depth = 32 
    output_depth = 1 
    INPUT = 'noise'
    pad = 'reflection'
    print('Sampling from CNN ...')
    samples = []

    net = get_net(input_depth, 'skip', pad,
                n_channels = 1,
                skip_n33d=1024, 
                skip_n33u=1024,
                skip_n11=4,
                num_scales=2,
                need_sigmoid = False,
                upsample_mode='bilinear').type(dtype)

    for i in range(num_samples):
        if i % 100 == 0:
            print("# %d" % i)
        net.apply(weight_reset)
        net_input = get_noise(input_depth, INPUT, (map_size, map_size)).type(dtype) 
        out = net(net_input)
        out_np = out.detach().cpu().numpy()[0]
        samples.append(out_np)
    samples = np.squeeze(np.array(samples))
    return samples


# load and save samples
def save_objs(filename, objs):
    np.save(filename, objs)

def load_objs(filename):
    return np.load(filename)


# compute kernel from samples 
# squared Euclidian distance 
def compute_mean_cov_from_samples(samples, target_size = 64):
    N, W, H = samples.shape
    max_dis = (target_size - 1) ** 2 * 2
    K_d = np.zeros(max_dis + 1) 
    count_d = np.zeros(max_dis + 1)
    mu = np.mean(samples)
    # take the center pixels to avoid the effect of padding
    for margin in range(int(W/2-target_size/2), int(W/2)):
        for ii in range(margin, margin+target_size):
            for jj in range(margin, margin+target_size):
                if ii < W and jj < W:
                    dist2 = (ii - margin) ** 2 + (jj - margin) ** 2
                    K_d[dist2] += np.sum((samples[:, margin, margin] - mu) * (samples[:, ii, jj] - mu))
                    count_d[dist2] += N
    for i in range(max_dis+1):
        if count_d[i] != 0:
            K_d[i] /= count_d[i]   
    return K_d, mu     


# estimate kernel from the samples
def compute_kernel(samples, size = 64):
    
    N, H, W = samples.shape
    samples = samples[:, int(H/2 - size/2) : int(H/2 + size/2), int(W/2-size/2):int(W/2 + size/2)]
    print(samples.shape)
    mean = samples.mean(axis=0).flatten()
    data = samples.reshape((N, size*size)) - mean
    kernel = np.cov(data, rowvar=0)
    # np_plot(kernel, 'kernel', opt='map')
    return kernel


# get big kernel from stationary kernel 
def convert_kernel(K_d, H = 64):
    K = np.zeros((H * H, H * H))
    for i in range(H):
    # print(i)
        for j in range(H):
            for p in range(H):
                for q in range(H):
                    aa = i*H + j
                    bb = p*H + q
                    dist2 = (p-i) ** 2 + (q - j) ** 2
                    K[aa, bb] = K_d[dist2]
    return K


# inpainting with non-stationary kernel
def GP_DIP_inpaint(img, mask, kernel):
    flat_mask = mask.flatten()
    flat_img = img.flatten()
    known_indices = np.nonzero(flat_mask)[0]
    unknown_indices = np.nonzero(np.logical_not(flat_mask))[0]

    y = flat_img[known_indices]
    k = kernel[known_indices, :][:, known_indices]
    k_star = kernel[known_indices, :][:, unknown_indices]

    k_inv = np.linalg.inv(k)
                                    
    mean = k_star.T.dot(k_inv).dot(y)

    masked_img = flat_img.copy()
    masked_img[unknown_indices] = 0
    masked_img = masked_img.reshape(*(img.shape))

    inpainted_img = flat_img.copy()
    inpainted_img[unknown_indices] = np.clip(mean, 0, 1)
    inpainted_img = inpainted_img.reshape(*(img.shape))
    return (inpainted_img, masked_img)


# draw kernel
def draw_Kernel_curve(K):
    plt.figure()
    if type(K).__module__ is not np.__name__:
        K_d = K.numpy()
    else:
        K_d = K
    idx = np.where(K_d != 0)[0]
    plt.plot(np.sqrt(idx), K_d[idx])
    plt.xlim((0, 100))
    plt.ylim((0, np.max(K_d)))
    plt.xlabel('d')
    plt.ylabel('K_d')



def RBF_baseline_kernel(scale = 3, lengthscale = 4.06):
    return scale * torch.exp(-0.5 * torch.linspace(0, 10000, 10000) / lengthscale)


