from STAC_AVC.utils_avc import read_image_resize_rect
from STAC_AVC.SAVC import SAVC
import numpy as np
import os
import matplotlib.pyplot as plt
from STAC_AVC.iqa.ssim_np import ssim_eval, msssim

import random

ssim_mod = lambda x, y: ssim_eval(x, y)
ssim_func = lambda x, y: -10*np.log10(ssim_mod(x, y))

msssim_mod = lambda x, y: msssim(x, y)
msssim_func = lambda x, y: -10*np.log10(msssim_mod(x, y))

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
        msssim_val = msssim_func(img1, img2)
        psnr = 10*np.log10(255**2/mse_val)
    else:
        weights = np.array([8, 0, 0])
        weights = weights/np.sum(weights)
        mse_score = np.zeros((3))
        ssim_score = np.zeros((3))
        msssim_score = np.zeros((3))

        for i in range(3):
            img_1_tmp = img1[:, :, i] 
            img_2_tmp = img2[:, :, i]
            mse_score[i] = np.mean(np.square(255*(img_1_tmp - img_2_tmp)/2))
            ssim_score[i] = ssim_func(img_1_tmp, img_2_tmp)
            msssim_score[i] = msssim_func(img_1_tmp, img_2_tmp)
        ssim_score[np.isnan(ssim_score)] = 0
        ssim_score[np.isinf(ssim_score)] = 0
        ssim_val = ssim_score @ weights
        mse_val = mse_score @ weights
        msssim_val = msssim_score @ weights
        psnr = 10*np.log10(255**2/mse_val)
    return psnr, ssim_val, msssim_val


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
num_images = 1
random.seed(0)
random.shuffle(dirs)
dirs = dirs[:num_images]

def main():
    psnr_vals = np.zeros((nqs, len(dirs)))
    ssim_vals = np.zeros_like(psnr_vals)
    msssim_vals = np.zeros_like(psnr_vals)

    bits = []

    for i in range(num_images):
        fullpath = os.path.join(path,dirs[i])
        img, depth = read_image_resize_rect(fullpath, true_N)
        img = img[:, :, 0]

        bits_img_savc = []

        for j in range(nqs):

            qual_idx = j
            comp_img_jpeg, bits_tmp = compress_AVC(qual_idx, img)
            bits_img_savc.append(bits_tmp)

            psnr_vals[j, i], ssim_vals[j, i], msssim_vals[j, i] = evaluate_metrics(img, comp_img_jpeg)

        bits.append(bits_img_savc)

        total_bits = np.array(bits_img_savc)/(img.shape[0]*img.shape[1])


import cProfile, pstats, io
from pstats import SortKey

pr = cProfile.Profile()
pr.enable()
main()
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.reverse_order().print_stats()
print(s.getvalue())
breakpoint()