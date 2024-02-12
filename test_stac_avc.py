from STAC_AVC.utils_avc import read_image_resize_rect
from STAC_AVC.SAVC import SAVC
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import random

ssim_mod = lambda x, y: 1-ssim(x, y, data_range=2)
ssim_func = lambda x, y: -10*np.log10(ssim_mod(x, y))

true_N = (128, 128)
nqs = 8
N = 8

flag_uniform = True

savc = SAVC(nqs, flag_uniform=flag_uniform)


def compress_AVC(qual_lev, img):
    res, Y, bits = savc.compress(img, qual_lev)
    return Y, bits


def evaluate_metrics(img1, img2):
    # now compute the other metrics for each channel
    img1 = 2*img1/255
    img2 = 2*img2/255

    if (len(img1.shape) == 2):
        mse_val = np.mean(np.square(255*(img1 - img2)/2))
        ssim_val = ssim_func(img1, img2)
        psnr = 10*np.log10(255**2/mse_val)
    else:
        weights = np.array([8, 0, 0])
        weights = weights/np.sum(weights)
        mse_score = np.zeros((3))
        ssim_score = np.zeros((3))

        for i in range(3):
            img_1_tmp = img1[:, :, i] 
            img_2_tmp = img2[:, :, i]
            mse_score[i] = np.mean(np.square(255*(img_1_tmp - img_2_tmp)/2))
            ssim_score[i] = ssim_func(img_1_tmp, img_2_tmp)
        ssim_score[np.isnan(ssim_score)] = 0
        ssim_score[np.isinf(ssim_score)] = 0
        ssim_val = ssim_score @ weights
        mse_val = mse_score @ weights
        psnr = 10*np.log10(255**2/mse_val)
    return psnr, ssim_val


def get_touples(mag, bits):
    arr_out = np.zeros((len(mag), 2))
    order = np.argsort(bits)
    arr_out[:, 0] = bits[order]
    arr_out[:, 1] = mag[order]
    return arr_out


def get_mean_format(data):
    mean = np.round(np.mean(data), 2)
    return '{}'.format(mean)


path = 'STAC_AVC/Images/CLIC/Testing/'
dirs = os.listdir(path)
num_images = 40
random.seed(0)
random.shuffle(dirs)
dirs = dirs[:num_images]


mse_vals = np.zeros((nqs, len(dirs)))
ssim_vals = np.zeros_like(mse_vals)
bits = []

for i in range(num_images):
    fullpath = os.path.join(path,dirs[i])
    img, depth = read_image_resize_rect(fullpath, true_N)
    img = img[:, :, 0]
    depth = 1

    savc.set_Q(img)
    bits_img_savc = []

    for j in range(nqs):

        qual_idx = j
        comp_img_jpeg, bits_tmp = compress_AVC(qual_idx, img)
        bits_img_savc.append(bits_tmp)

        mse_vals[j, i], ssim_vals[j, i] = evaluate_metrics(img, comp_img_jpeg)

    bits.append(bits_img_savc)

    total_bits = np.array(bits_img_savc)/(img.shape[0]*img.shape[1])

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(total_bits, mse_vals[:, i], label='AVC',linewidth=3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.xlabel('Bits per pixel', fontsize=16)
    plt.title('MSE', fontsize=16)
    plt.subplot(1, 2, 2)
    plt.plot(total_bits, ssim_vals[:, i], label='AVC', linewidth=3)
    plt.xlabel('Bits per pixel', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize = 16)
    plt.title('SSIM', fontsize=16)
    plt.tight_layout()
    str_out = 'sketch/simul_tmps/{}.pdf'.format(i)
    plt.show()