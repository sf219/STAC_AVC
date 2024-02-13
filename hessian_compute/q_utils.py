from STAC_AVC.iqa.brisque_jax import brisque_loss
from STAC_AVC.iqa.niqe_jax import niqe_loss
import jax.numpy as jnp
import jax
from STAC_AVC.lpips_jax_master.lpips_jax.lpips import LPIPSEvaluator
from functools import partial
from jax.scipy.stats import norm
import numpy as np
from scipy.signal import convolve2d
import scipy.stats


# the values of these parameters are very temporary 
def localvar(f):
    # Set default values
    # this default values are chosen based on scikit image implementation
    sigma = 1.75
    truncate = 3.5
    r = int(truncate * sigma + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1

    x = np.linspace(-truncate, truncate, win_size)
    window = scipy.stats.norm.pdf(x, scale=sigma) * scipy.stats.norm.pdf(x[:, None], scale=sigma)
    window = window/np.sum(window)

    f = np.pad(f, r, mode='symmetric')

    ux = convolve2d(f, window, mode='valid')
    uxx = convolve2d(f**2, window, mode='valid')
    
    # Compute local variance
    vx = uxx - ux**2
    return vx


def get_perceptual_q(f_train, quant, N=8, data_range=2):
    # weights for SSIM, from error model with sum(q) constraint. The closed
    # form is obtained from Lagrange multiplier trick
    c2 = (0.03*data_range)**2
    vx = localvar(f_train)
    gm = quant / np.sqrt(12*(2*vx + c2))
    qx = (gm.size + np.sum(np.power(gm, 2)))/np.sum(gm)*gm - np.power(gm, 2)
    return qx, vx


@jax.jit
def block_multiply(img, mtx_trans):
    img = img.squeeze()
    N = np.sqrt(mtx_trans.shape[0]).astype(int)
    img_trans = jnp.zeros_like(img)
    mtx_trans = jnp.array(mtx_trans)

    def multiply_by_mtx(img_block):
        img_block = img_block.ravel('F')
        return (mtx_trans @ img_block).reshape((N, N), order='F')
    
    matrix_fun = jax.jit(jax.vmap(multiply_by_mtx))
    img_shape = img.shape
    # reshape img into blocks of N x N
    img = reshape_image_blocks(img, N)
    # apply the transformation to each block
    img_trans = matrix_fun(img)
    img_trans = invert_reshape_image_blocks(img_trans, img_shape, N)
    img_trans = jnp.reshape(img_trans, (1, 1, img_trans.shape[0], img_trans.shape[1]))
    return img_trans


def to_tensor(img):
    img = jnp.array(img).reshape((1, 1, img.shape[0], img.shape[1]))
    return img


def hvp(f, primals, tangents):
    return jax.jvp(jax.grad(f), primals, tangents)[1]


def sample_hessian(func, img, sampler):
    fun_samp = lambda x: func(img, x)
    hessian = hvp(fun_samp, (img,), (sampler,))
    return hessian


def reshape_image_blocks(image, block_size):
    # Get the shape of the original image
    height, width = image.shape

    # Calculate the number of blocks in both dimensions
    num_blocks_height = height // block_size
    num_blocks_width = width // block_size

    # Reshape the image into blocks
    reshaped_image = jnp.reshape(image[:num_blocks_height * block_size, :num_blocks_width * block_size],
                                  (num_blocks_height, block_size, num_blocks_width, block_size))

    # Transpose to have blocks in the first and third dimensions
    reshaped_image = jnp.transpose(reshaped_image, (0, 2, 1, 3))

    # Reshape to get the final result
    reshaped_image = jnp.reshape(reshaped_image, (num_blocks_height * num_blocks_width, block_size, block_size))

    return reshaped_image

def invert_reshape_image_blocks(reshaped_image, original_shape, block_size):
    # Get the shape of the original image
    num_blocks, _, _ = reshaped_image.shape

    # Calculate the number of blocks in both dimensions
    num_blocks_height, num_blocks_width = original_shape[0] // block_size, original_shape[1] // block_size

    # Reshape back to 4D array
    reshaped_image = reshaped_image.reshape((num_blocks_height, num_blocks_width, block_size, block_size))

    # Transpose to have blocks in the first and third dimensions
    reshaped_image = jnp.transpose(reshaped_image, (0, 2, 1, 3))

    # Reshape back to the original image
    original_image = reshaped_image.reshape((num_blocks_height * block_size, num_blocks_width * block_size))

    return original_image


@jax.jit
def brisque_func(img1, img2):
    bris_1 = brisque_loss(img1)
    bris_2 = brisque_loss(img2)
    diff = (bris_2 - bris_1)
    cdf = norm.cdf(diff) - 0.1
    term_2 = diff**2 * cdf
    return term_2


@jax.jit
def niqe_func(img1, img2):
    bris_1 = niqe_loss(img1)
    bris_2 = niqe_loss(img2)
    diff = -(bris_2 - bris_1)
    cdf = norm.cdf(diff) - 0.1
    term_2 = diff**2 * cdf
    return term_2

lp_jax = LPIPSEvaluator(replicate=False)

@jax.jit
def compute_LPIPS_gs(img1, img2):
    img1 = img1.squeeze()
    img2 = img2.squeeze()
    true_N = img1.shape
    img1 = jnp.repeat(img1[:, :, jnp.newaxis], 3, axis=2)
    img1 = img1.transpose(2, 0, 1)
    img1 = img1.reshape(1, 3, true_N[0], true_N[1])
    img1 = jnp.array(img1).transpose(0, 2, 3, 1)
    img2 = jnp.repeat(img2[:, :, jnp.newaxis], 3, axis=2)
    img2 = img2.transpose(2, 0, 1)
    img2 = img2.reshape(1, 3, true_N[0], true_N[1])
    img2 = jnp.array(img2).transpose(0, 2, 3, 1)
    lpips = lp_jax(img1, img2)
    lpips = lpips[0][0][0][0]
    return lpips


@jax.jit
def compute_LPIPS_color(img1, img2):
    img1 = img1.squeeze()
    img2 = img2.squeeze()
    true_N = img1.shape
    img1 = img1.transpose(2, 0, 1)
    img1 = img1.reshape(1, 3, true_N[0], true_N[1])
    img1 = jnp.array(img1).transpose(0, 2, 3, 1)
    img2 = img2.transpose(2, 0, 1)
    img2 = img2.reshape(1, 3, true_N[0], true_N[1])
    img2 = jnp.array(img2).transpose(0, 2, 3, 1)
    lpips = lp_jax(img1, img2)
    lpips = lpips[0][0][0][0]
    return lpips
