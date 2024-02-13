from STAC_AVC.utils_avc import read_image_resize_rect, ycbcr2rgb, bdrate

from STAC_AVC.SAVC import SAVC
from STAC_AVC.IAGFT_AVC import IAGFT_AVC
from STAC_AVC.hessian_compute.compute_Q_jax import compute_Q_ssim as compute_Q
from STAC_AVC.hessian_compute.q_ops_gain_shape import q_ops_ssim as q_ops

import numpy as np
import os
import matplotlib.pyplot as plt
from STAC_AVC.iqa.jax_ssim import jax_ssim_eval, jax_msssim
from STAC_AVC.iqa.brisque_jax import brisque_loss
from STAC_AVC.hessian_compute.q_utils import compute_LPIPS_gs, compute_LPIPS_color
import jax.numpy as jnp
import random

ssim_mod = lambda x, y: jax_ssim_eval(x, y)
ssim_func = lambda x, y: -10*np.log10(ssim_mod(x, y))

msssim_mod = lambda x, y: jax_msssim(x, y)
msssim_func = lambda x, y: -10*np.log10(msssim_mod(x, y))

true_N = (512, 512)
nqs = 8
N = 4

flag_uniform = True

compute_q_obj = compute_Q(true_N=true_N, sampling_depth=16)
q_ops_obj = q_ops(true_N=true_N, N=N, nqs=nqs)
savc = SAVC(nqs, flag_uniform=flag_uniform)
qavc = IAGFT_AVC(nqs, compute_Q_obj=compute_q_obj, q_ops_obj=q_ops_obj, flag_uniform=flag_uniform)


def compress_AVC(qual_lev, img):
    res, Y, bits = savc.compress(img, qual_lev)
    return Y, bits


def compress_QAVC(qual_lev, img):
    res, Y, bits = qavc.compress(img, qual_lev)
    return Y, bits


def evaluate_metrics(img1, img2):
    # now compute the other metrics for each channel
    
    img1_norm = 2*img1/255
    img2_norm = 2*img2/255

    if (len(img1.shape) == 2):
        mse_val = np.mean(np.square(255*(img1_norm - img2_norm)/2))
        ssim_val = ssim_func(img1_norm, img2_norm)
        msssim_val = msssim_func(img1_norm, img2_norm)
        psnr = 10*np.log10(255**2/mse_val)
        img2_tensor = jnp.array(img2_norm).reshape((1, 1, img2_norm.shape[0], img2_norm.shape[1]))
        img1_tensor = jnp.array(img1_norm).reshape((1, 1, img1_norm.shape[0], img1_norm.shape[1]))
        brisque = brisque_loss(img2_tensor)
        lpips = -10*np.log10(compute_LPIPS_gs(img1_tensor, img2_tensor))
    else:
        weights = np.array([8, 0, 0])
        weights = weights/np.sum(weights)
        mse_score = np.zeros((3))
        ssim_score = np.zeros((3))
        msssim_score = np.zeros((3))
        brisque_score = np.zeros((3))

        # first compute the LPIPS for the full image
        img_1_rgb = ycbcr2rgb(img1)
        img_2_rgb = ycbcr2rgb(img2)

        img_1_rgb = 2*img_1_rgb/255-1
        img_2_rgb = 2*img_2_rgb/255-1

        ten_img_1_rgb = jnp.array(img_1_rgb)
        ten_img_2_rgb = jnp.array(img_2_rgb)

        lpips = -10*np.log10(compute_LPIPS_color(ten_img_1_rgb, ten_img_2_rgb))
        for i in range(3):
            img_1_tmp = img1_norm[:, :, i] 
            img_2_tmp = img2_norm[:, :, i]
            mse_score[i] = np.mean(np.square(255*(img_1_tmp - img_2_tmp)/2))
            ssim_score[i] = ssim_func(img_1_tmp, img_2_tmp)
            msssim_score[i] = msssim_func(img_1_tmp, img_2_tmp)
            brisque_score[i] = brisque_loss(jnp.array(img_2_tmp).reshape((1, 1, img_2_tmp.shape[0], img_2_tmp.shape[1])))
        ssim_score[np.isnan(ssim_score)] = 0
        ssim_score[np.isinf(ssim_score)] = 0
        ssim_val = ssim_score @ weights
        mse_val = mse_score @ weights
        msssim_val = msssim_score @ weights
        brisque = brisque_score @ weights
        psnr = 10*np.log10(255**2/mse_val)
    return psnr, ssim_val, msssim_val, brisque, lpips


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

psnr_vals = np.zeros((nqs, len(dirs)))
ssim_vals = np.zeros_like(psnr_vals)
msssim_vals = np.zeros_like(psnr_vals)
brisque_vals = np.zeros_like(psnr_vals)
lpips_vals = np.zeros_like(psnr_vals)

psnr_vals_qavc = np.zeros((nqs, len(dirs)))
ssim_vals_qavc = np.zeros_like(psnr_vals_qavc)
msssim_vals_qavc = np.zeros_like(psnr_vals_qavc)
brisque_vals_qavc = np.zeros_like(psnr_vals_qavc)
lpips_vals_qavc = np.zeros_like(psnr_vals_qavc)

bdrate_psnr = np.zeros((len(dirs)))
bdrate_ssim = np.zeros((len(dirs)))
bdrate_msssim = np.zeros((len(dirs)))
bdrate_brisque = np.zeros((len(dirs)))
bdrate_lpips = np.zeros((len(dirs)))

bits = []
bits_qavc = []

for i in range(num_images):
    fullpath = os.path.join(path,dirs[i])
    img, depth = read_image_resize_rect(fullpath, true_N)
    img = img[:, :, 0]
    depth = 1

    bits_img_savc = []
    bits_img_qavc = []

    qavc.set_Q(img)

    for j in range(nqs):

        qual_idx = j
        comp_img_jpeg, bits_tmp = compress_AVC(qual_idx, img)
        comp_img_jpeg_qavc, bits_tmp_qavc = compress_QAVC(qual_idx, img)
        bits_img_savc.append(bits_tmp)
        bits_img_qavc.append(bits_tmp_qavc)

        psnr_vals[j, i], ssim_vals[j, i], msssim_vals[j, i], brisque_vals[j, i], lpips_vals[j, i] = evaluate_metrics(img, comp_img_jpeg)
        psnr_vals_qavc[j, i], ssim_vals_qavc[j, i], msssim_vals_qavc[j, i], brisque_vals_qavc[j, i], lpips_vals_qavc[j, i] = evaluate_metrics(img, comp_img_jpeg_qavc)

    bits.append(bits_img_savc)
    bits_qavc.append(bits_img_qavc)

    total_bits = np.array(bits_img_savc)/(img.shape[0]*img.shape[1])
    total_bits_qavc = np.array(bits_img_qavc)/(img.shape[0]*img.shape[1])

    tuple_psnr_avc = get_touples(psnr_vals[:, i], total_bits)
    tuple_psnr_qavc = get_touples(psnr_vals_qavc[:, i], total_bits_qavc)

    tuple_ssim_avc = get_touples(ssim_vals[:, i], total_bits)
    tuple_ssim_qavc = get_touples(ssim_vals_qavc[:, i], total_bits_qavc)

    tuple_msssim_avc = get_touples(msssim_vals[:, i], total_bits)
    tuple_msssim_qavc = get_touples(msssim_vals_qavc[:, i], total_bits_qavc)

    tuple_brisque_avc = get_touples(brisque_vals[:, i], total_bits)
    tuple_brisque_qavc = get_touples(brisque_vals_qavc[:, i], total_bits_qavc)

    tuple_lpips_avc = get_touples(lpips_vals[:, i], total_bits)
    tuple_lpips_qavc = get_touples(lpips_vals_qavc[:, i], total_bits_qavc)

    bdrate_psnr[i] = bdrate(tuple_psnr_avc, tuple_psnr_qavc)
    bdrate_ssim[i] = bdrate(tuple_ssim_avc, tuple_ssim_qavc)
    bdrate_msssim[i] = bdrate(tuple_msssim_avc, tuple_msssim_qavc)
    bdrate_brisque[i] = bdrate(tuple_brisque_avc, tuple_brisque_qavc)
    bdrate_lpips[i] = bdrate(tuple_lpips_avc, tuple_lpips_qavc)

    qavc_psnr = get_mean_format(bdrate_psnr[i])
    qavc_ssim = get_mean_format(bdrate_ssim[i])
    qavc_msssim = get_mean_format(bdrate_msssim[i])
    qavc_brisque = get_mean_format(bdrate_brisque[i])
    qavc_lpips = get_mean_format(bdrate_lpips[i])

    print('Image: ', i, 'PSNR: ', qavc_psnr, 'SSIM: ', qavc_ssim, 'MS-SSIM: ', qavc_msssim, 'BRISQUE: ', qavc_brisque, 'LPIPS: ', qavc_lpips)

    qavc_psnr = get_mean_format(bdrate_psnr[:i+1])
    qavc_ssim = get_mean_format(bdrate_ssim[:i+1])
    qavc_msssim = get_mean_format(bdrate_msssim[:i+1])
    qavc_brisque = get_mean_format(bdrate_brisque[:i+1])
    qavc_lpips = get_mean_format(bdrate_lpips[:i+1])

    print('Mean iter: ', i, 'PSNR: ', qavc_psnr, 'SSIM: ', qavc_ssim, 'MS-SSIM: ', qavc_msssim, 'BRISQUE: ', qavc_brisque, 'LPIPS: ', qavc_lpips)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 5, 1)
    plt.plot(total_bits, psnr_vals[:, i], label='AVC',linewidth=3)
    plt.plot(total_bits_qavc, psnr_vals_qavc[:, i], label='QAVC',linewidth=3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.xlabel('Bits per pixel', fontsize=16)
    plt.title('PSNR', fontsize=16)
    plt.subplot(1, 5, 2)
    plt.plot(total_bits, ssim_vals[:, i], label='AVC', linewidth=3)
    plt.plot(total_bits_qavc, ssim_vals_qavc[:, i], label='QAVC', linewidth=3)
    plt.xlabel('Bits per pixel', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize = 16)
    plt.title('SSIM', fontsize=16)
    plt.subplot(1, 5, 3)
    plt.plot(total_bits, msssim_vals[:, i], label='AVC', linewidth=3)
    plt.plot(total_bits_qavc, msssim_vals_qavc[:, i], label='QAVC', linewidth=3)
    plt.xlabel('Bits per pixel', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.title('MS-SSIM', fontsize=16)
    plt.subplot(1, 5, 4)
    plt.plot(total_bits, brisque_vals[:, i], label='AVC', linewidth=3)
    plt.plot(total_bits_qavc, brisque_vals_qavc[:, i], label='QAVC', linewidth=3)
    plt.xlabel('Bits per pixel', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.title('BRISQUE', fontsize=16)
    plt.subplot(1, 5, 5)
    plt.plot(total_bits, lpips_vals[:, i], label='AVC', linewidth=3)
    plt.plot(total_bits_qavc, lpips_vals_qavc[:, i], label='QAVC', linewidth=3)
    plt.xlabel('Bits per pixel', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.title('LPIPS', fontsize=16)
    plt.tight_layout()
    str_out = 'sketch/simul_tmps/{}.pdf'.format(i)
    plt.show()