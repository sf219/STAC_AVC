import scipy 
from skimage.metrics import structural_similarity as ssim_sk
import numpy as np

# these functions can be optimizing by exploiting the fact that the Gaussian filters are separable

def preprocess(img, win_size=9):
    pad_int = (win_size-1)//2
    img = np.pad(img, pad_int, mode='symmetric')
    return img


def ssim(img1, img2):
    img1 = img1.squeeze()
    img2 = img2.squeeze()
    mssim, _ = _ssim(img1, img2)
    return 1-mssim


def ssim_eval(img1, img2):
    img1 = img1.squeeze()
    img2 = img2.squeeze()
    F = np.maximum(1, img1.shape[0]//256)
    kernel = np.ones((F, F))
    kernel = kernel/np.sum(kernel)
    img1 = scipy.signal.convolve(img1, kernel, mode='valid')[::F, ::F]
    img2 = scipy.signal.convolve(img2, kernel, mode='valid')[::F, ::F]
    img1 = np.array(img1)
    img2 = np.array(img2)
    mssim = ssim_sk(img1, img2, data_range=2, gaussian_weights=True)
    return 1-mssim

def just_cs(img1, img2):

    # compute input parameters
    c2 = (0.03*2)**2
    truncate = 3.5
    sigma = 1.5
    r = int(truncate * sigma + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    
    img1 = img1.squeeze()
    img2 = img2.squeeze()

    pad = (win_size-1)//2
    img1 = np.pad(img1, pad, mode='symmetric')
    img2 = np.pad(img2, pad, mode='symmetric')

    x = np.linspace(-truncate, truncate, win_size)
    window = scipy.stats.norm.pdf(x, scale=sigma) * scipy.stats.norm.pdf(x[:, None], scale=sigma)
    window = window/np.sum(window)
    
    mu_ref = scipy.signal.convolve(img1, window, mode='valid')
    mu2_ref = scipy.signal.convolve(np.square(img1), window, mode='valid')
    mu_tar = scipy.signal.convolve(img2, window, mode='valid')
    mu2_tar = scipy.signal.convolve(np.square(img2), window, mode='valid')
    cross = scipy.signal.convolve(img1 * img2, window, mode='valid') - mu_ref * mu_tar

    mu_ref2 = np.square(mu_ref)
    mu_tar2 = np.square(mu_tar)

    sigma2_ref = mu2_ref - mu_ref2
    sigma2_tar = mu2_tar - mu_tar2
    
    cs_map = (2*cross + c2)/(sigma2_ref + sigma2_tar + c2)
    return cs_map.mean()

def _ssim(img1, img2):

    # compute input parameters
    c2 = (0.03*2)**2
    c1 = (0.01*2)**2
    truncate = 3.5
    sigma = 1.5
    r = int(truncate * sigma + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    
    img1 = img1.squeeze()
    img2 = img2.squeeze()

    pad = (win_size-1)//2
    img1 = np.pad(img1, pad, mode='symmetric')
    img2 = np.pad(img2, pad, mode='symmetric')

    x = np.linspace(-truncate, truncate, win_size)
    window = scipy.stats.norm.pdf(x, scale=sigma) * scipy.stats.norm.pdf(x[:, None], scale=sigma)
    window = window/np.sum(window)
    
    mu_ref = scipy.signal.convolve(img1, window, mode='valid')
    mu2_ref = scipy.signal.convolve(np.square(img1), window, mode='valid')
    mu_tar = scipy.signal.convolve(img2, window, mode='valid')
    mu2_tar = scipy.signal.convolve(np.square(img2), window, mode='valid')
    cross = scipy.signal.convolve(img1 * img2, window, mode='valid') - mu_ref * mu_tar

    mu_ref2 = np.square(mu_ref)
    mu_tar2 = np.square(mu_tar)

    sigma2_ref = mu2_ref - mu_ref2
    sigma2_tar = mu2_tar - mu_tar2
    
    cs_map = (2*cross + c2)/(sigma2_ref + sigma2_tar + c2)
    ssim_img = (2*mu_ref*mu_tar + c1)/(mu_ref2 + mu_tar2 + c1) * cs_map

    mssim = np.mean(ssim_img)
    cs = np.mean(cs_map)
    return mssim, cs


def msssim(img1, img2):
    img1 = img1.squeeze()
    img2 = img2.squeeze()
    downsample_filter = np.ones((2, 2))/4.0
    levels = 5
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    msssim = np.array([])
    for i in range(levels):
        if (i < levels - 1):
            cs = just_cs(img1, img2)
            msssim = np.append(msssim, cs)
            if (img1.shape[0] % 2 == 1):
                img1 = np.pad(img1, (1, 1), mode='reflect')
                img2 = np.pad(img2, (1, 1), mode='reflect')
            img1 = scipy.signal.convolve(img1, downsample_filter, mode='valid')[::2, ::2]
            img2 = scipy.signal.convolve(img2, downsample_filter, mode='valid')[::2, ::2]
        else:
            ssim_channel, _ = _ssim(img1, img2)
    ssim_per_channel = np.maximum(ssim_channel, 0)
    mcs_and_ssim = np.append(msssim, ssim_per_channel)
    ms_ssim_val = np.prod(mcs_and_ssim**weights)
    return 1 - np.mean(ms_ssim_val)
