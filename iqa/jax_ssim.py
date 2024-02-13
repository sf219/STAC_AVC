import jax.scipy as jsp
import jax
import jax.numpy as jnp
from skimage.metrics import structural_similarity as ssim
import numpy as np

def preprocess(img, win_size=9):
    pad_int = (win_size-1)//2
    img = jnp.pad(img, pad_int, mode='symmetric')
    return img


@jax.jit
def jax_ssim(img1, img2):
    img1 = img1.squeeze()
    img2 = img2.squeeze()
    mssim, _ = _jax_ssim(img1, img2)
    return 1-mssim


def jax_ssim_eval(img1, img2):
    img1 = img1.squeeze()
    img2 = img2.squeeze()
    F = jnp.maximum(1, img1.shape[0]//256)
    kernel = jnp.ones((F, F))
    kernel = kernel/jnp.sum(kernel)
    img1 = jsp.signal.convolve(img1, kernel, mode='valid')[::F, ::F]
    img2 = jsp.signal.convolve(img2, kernel, mode='valid')[::F, ::F]
    img1 = np.array(img1)
    img2 = np.array(img2)
    mssim = ssim(img1, img2, data_range=2, gaussian_weights=True)
    return 1-mssim

@jax.jit
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
    img1 = jnp.pad(img1, pad, mode='symmetric')
    img2 = jnp.pad(img2, pad, mode='symmetric')

    x = jnp.linspace(-truncate, truncate, win_size)
    window = jsp.stats.norm.pdf(x, scale=sigma) * jsp.stats.norm.pdf(x[:, None], scale=sigma)
    window = window/jnp.sum(window)
    
    mu_ref = jsp.signal.convolve(img1, window, mode='valid')
    mu2_ref = jsp.signal.convolve(jnp.square(img1), window, mode='valid')
    mu_tar = jsp.signal.convolve(img2, window, mode='valid')
    mu2_tar = jsp.signal.convolve(jnp.square(img2), window, mode='valid')
    cross = jsp.signal.convolve(img1 * img2, window, mode='valid') - mu_ref * mu_tar

    mu_ref2 = jnp.square(mu_ref)
    mu_tar2 = jnp.square(mu_tar)

    sigma2_ref = mu2_ref - mu_ref2
    sigma2_tar = mu2_tar - mu_tar2
    
    cs_map = (2*cross + c2)/(sigma2_ref + sigma2_tar + c2)
    return cs_map.mean()


@jax.jit
def _jax_ssim(img1, img2):

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
    img1 = jnp.pad(img1, pad, mode='symmetric')
    img2 = jnp.pad(img2, pad, mode='symmetric')

    x = jnp.linspace(-truncate, truncate, win_size)
    window = jsp.stats.norm.pdf(x, scale=sigma) * jsp.stats.norm.pdf(x[:, None], scale=sigma)
    window = window/jnp.sum(window)
    
    mu_ref = jsp.signal.convolve(img1, window, mode='valid')
    mu2_ref = jsp.signal.convolve(jnp.square(img1), window, mode='valid')
    mu_tar = jsp.signal.convolve(img2, window, mode='valid')
    mu2_tar = jsp.signal.convolve(jnp.square(img2), window, mode='valid')
    cross = jsp.signal.convolve(img1 * img2, window, mode='valid') - mu_ref * mu_tar

    mu_ref2 = jnp.square(mu_ref)
    mu_tar2 = jnp.square(mu_tar)

    sigma2_ref = mu2_ref - mu_ref2
    sigma2_tar = mu2_tar - mu_tar2
    
    cs_map = (2*cross + c2)/(sigma2_ref + sigma2_tar + c2)
    ssim_img = (2*mu_ref*mu_tar + c1)/(mu_ref2 + mu_tar2 + c1) * cs_map

    mssim = jnp.mean(ssim_img)
    cs = jnp.mean(cs_map)
    return mssim, cs


@jax.jit
def jax_msssim(img1, img2):
    img1 = img1.squeeze()
    img2 = img2.squeeze()
    downsample_filter = jnp.ones((2, 2))/4.0
    levels = 5
    weights = jnp.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    msssim = jnp.array([])
    for i in range(levels):
        if (i < levels - 1):
            cs = just_cs(img1, img2)
            msssim = jnp.append(msssim, cs)
            if (img1.shape[0] % 2 == 1):
                img1 = jnp.pad(img1, (1, 1), mode='reflect')
                img2 = jnp.pad(img2, (1, 1), mode='reflect')
            img1 = jsp.signal.convolve(img1, downsample_filter, mode='valid')[::2, ::2]
            img2 = jsp.signal.convolve(img2, downsample_filter, mode='valid')[::2, ::2]
        else:
            ssim_channel, _ = _jax_ssim(img1, img2)
    ssim_per_channel = jnp.maximum(ssim_channel, 0)
    mcs_and_ssim = jnp.append(msssim, ssim_per_channel)
    ms_ssim_val = jnp.prod(mcs_and_ssim**weights)
    return 1 - jnp.mean(ms_ssim_val)


@jax.jit
def _jax_ssim_zp(img1, img2):

    # compute input parameters
    c2 = (0.03*2)**2
    c1 = (0.01*2)**2
    truncate = 3.5
    sigma = 1.5
    r = int(truncate * sigma + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    
    img1 = img1.squeeze()
    img2 = img2.squeeze()

    x = jnp.linspace(-truncate, truncate, win_size)
    window = jsp.stats.norm.pdf(x, scale=sigma) * jsp.stats.norm.pdf(x[:, None], scale=sigma)
    window = window/jnp.sum(window)
    
    mu_ref = jsp.signal.convolve2d(img1, window, mode='same', boundary='fill')
    mu2_ref = jsp.signal.convolve2d(jnp.square(img1), window, mode='same', boundary='fill')
    mu_tar = jsp.signal.convolve2d(img2, window, mode='same', boundary='fill')
    mu2_tar = jsp.signal.convolve2d(jnp.square(img2), window, mode='same', boundary='fill')
    cross = jsp.signal.convolve2d(img1 * img2, window, mode='same', boundary='fill') - mu_ref * mu_tar

    mu_ref2 = jnp.square(mu_ref)
    mu_tar2 = jnp.square(mu_tar)

    sigma2_ref = mu2_ref - mu_ref2
    sigma2_tar = mu2_tar - mu_tar2
    
    cs_map = (2*cross + c2)/(sigma2_ref + sigma2_tar + c2)
    ssim_img = (2*mu_ref*mu_tar + c1)/(mu_ref2 + mu_tar2 + c1) * cs_map

    mssim = jnp.mean(ssim_img)
    cs = jnp.mean(cs_map)
    return mssim, cs


@jax.jit
def jax_ssim_zp(img1, img2):
    mssim, _ = _jax_ssim_zp(img1, img2)
    return 1-mssim
