import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from typing import Sequence
from functools import partial
from jax.scipy.special import gammaln



def _gamma(x):
    """
    There's no gamma function in JAX, so we use the log abs gamma function and exp function instead.
    """
    return jnp.exp(jsp.special.gammaln(x))


def _estimate_aggd_param(x) -> Sequence[jnp.ndarray]:
    x = jnp.reshape(x, (1, 1, x.shape[0], x.shape[1]))
    gamma = jnp.arange(start=0.2, stop=10.001, step=0.001)
    r_table = jnp.exp(2 * gammaln(2. / gamma) - gammaln(1. / gamma) - gammaln(3. / gamma))
    r_table = jnp.tile(r_table, (x.shape[0], 1))

    mask_left = x < 0
    mask_right = x > 0
    count_left = mask_left.sum(axis=(-1, -2))
    count_right = mask_right.sum(axis=(-1, -2))

    left_sigma = jnp.sqrt((jnp.square(x * mask_left)).sum(axis=(-1, -2)) / count_left)
    right_sigma = jnp.sqrt((jnp.square(x * mask_right)).sum(axis=(-1, -2)) / count_right)

    gamma_hat = left_sigma / right_sigma
    ro_hat = jnp.square(jnp.abs(x).mean(axis=(-1, -2))) / jnp.mean(jnp.square(x), axis=(-1, -2))
    ro_hat_norm = (ro_hat * (gamma_hat**3 + 1) * (gamma_hat + 1)) / (gamma_hat**2 + 1)**2

    indexes = jnp.argmin(jnp.abs(ro_hat_norm - r_table), axis=-1)
    solution = gamma[indexes]
    return solution, left_sigma.squeeze(axis=-1), right_sigma.squeeze(axis=-1)


def _compute_feature(block):
    block = block.squeeze()

    alpha_gdd, beta_l, beta_r = _estimate_aggd_param(block)

    shifts = jnp.array([[0, 1], [1, 0], [1, 1], [1, -1]])

    shifts = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    features = [alpha_gdd, (beta_l+beta_r)/2]
    
    for shift in shifts:
        shifted_luma_nrmlzd = jnp.roll(block, shift, axis=(-2, -1))
        alpha, sigma_l, sigma_r = _estimate_aggd_param(block * shifted_luma_nrmlzd)
        eta = (sigma_r - sigma_l) * jnp.exp(
            gammaln(2. / alpha) - (gammaln(1. / alpha) + gammaln(3. / alpha)) / 2)
        features.extend((alpha, eta, sigma_l**2, sigma_r**2))
    # i want to put into an array alpha_gdd, sigma_gdd, and tmp_feat
    return jnp.stack(features, axis=-1).squeeze()


def _nancov(x):
    """
    Exclude whole rows that contain NaNs.
    """
    nan_cond = ~jnp.any(jnp.isnan(x), axis=1, keepdims=True)
    n = jnp.sum(nan_cond)

    x_filtered = jnp.where(nan_cond, x, jnp.zeros_like(x))
    x_mean = jnp.sum(x_filtered, axis=0) / n
    x_centered = jnp.where(nan_cond, x - x_mean, jnp.zeros_like(x))
    cov = jnp.matmul(x_centered.T, x_centered) / (n - 1)
    return cov


def _ggd_parameters(x):
    x = jnp.reshape(x, (1, 1, x.shape[0], x.shape[1]))
    gamma = jnp.arange(start=0.2, stop=10.001, step=0.001)
    r_table = jnp.exp(gammaln(1. / gamma) + gammaln(3. / gamma) - 2 * gammaln(2. / gamma))
    r_table = jnp.tile(r_table, (x.shape[0], 1))

    sigma_sq = jnp.square(x).mean(axis=(-1, -2))
    sigma = jnp.sqrt(sigma_sq)

    E = jnp.abs(x).mean(axis=(-1, -2))
    rho = sigma_sq / jnp.square(E)

    indexes = jnp.argmin(jnp.abs(rho - r_table), axis=-1)
    solution = gamma[indexes]
    return solution, sigma.squeeze(axis=-1)


def reshape_image_blocks(image, block_size):
    # Get the shape of the original image
    image = image.squeeze()
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

    reshaped_image = reshaped_image.reshape((num_blocks_height * num_blocks_width, block_size, block_size, 1))
    return reshaped_image


def _calculate_niqe(img, mu_pris_param, cov_pris_param, gaussian_window, block_size=96):
    h, w = img.shape
    img = img.reshape((h, w, 1))
    n_blocks_h = int(np.ceil(h / block_size))
    n_blocks_w = int(np.ceil(w / block_size))
    pad_h = (n_blocks_h * block_size - h)//2
    pad_w = (n_blocks_w * block_size - w)//2
    img = jnp.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='symmetric')
    img = img[jnp.newaxis, :n_blocks_h * block_size, :n_blocks_w * block_size].astype(jnp.float32)  # TODO: Fix later
    gaussian_window = _gaussian_filter()
    k_h, k_w = gaussian_window.shape
    distparams = []
    for scale in (1, 2):
        img_tmp = img.squeeze()
        gaussian_window = gaussian_window.reshape(k_h, k_w)
        img_pad = jnp.pad(img_tmp, ((k_h // 2, k_h // 2), (k_w // 2, k_w // 2)), mode='symmetric')
        mu = jsp.signal.convolve(img_pad, gaussian_window, mode='valid')
        lf_tmp = (img_tmp - mu)**2
        lf_tmp = jnp.pad(lf_tmp, ((k_h // 2, k_h // 2), (k_w // 2, k_w // 2)), mode='symmetric')
        var = jsp.signal.convolve(lf_tmp, gaussian_window, mode='valid')

        sigma = jnp.sqrt(var + 0.25)
        img_norm = (img_tmp - mu) / (sigma + 0.5)

        img_tmp = img_norm.reshape((img_norm.shape[0], img_norm.shape[1], 1))
        img_norm = reshape_image_blocks(img_tmp, block_size // scale)
        #einops.rearrange(  # blocks are arranged from w to h. (w h) b1 b2 c
        #    img_norm, '(h b1) (w b2) c -> (w h) b1 b2 c', b1=block_size // scale, b2=block_size // scale)
        dist_vals = jax.vmap(_compute_feature)(img_norm)
        if scale == 1:
            img = jax.image.resize(img, (1, img.shape[1] // 2, img.shape[2] // 2, 1), method='cubic')
        distparams.append(jnp.asarray(dist_vals))
    distparams_vec = jnp.concatenate(distparams, axis=1)
    #distparams_vec = distparams_vec.T
    #distparams = jax.vmap(_scale_features)(distparams)
    

    mu_dist_param = jnp.nanmean(distparams_vec, axis=0, keepdims=True)
    #x_centered = distparams - mu_dist_param
    #cov_dist_param = jnp.matmul(x_centered.T, x_centered) / (distparams.shape[0] - 1)
    cov_dist_param = _nancov(distparams_vec)

    invcov_pris = jnp.linalg.inv(cov_pris_param)
    mix_mtx = (jnp.eye(cov_pris_param.shape[0]) + invcov_pris @ cov_dist_param)/2
    invcov_dist_params = jax.lax.stop_gradient(jnp.linalg.pinv(mix_mtx, rcond=1e-15))
    invcov_dist_params = invcov_dist_params @ invcov_pris
    #invcov_dist_params = jnp.linalg.pinv((cov_pris_param), rcond=1e-15)
    val = jnp.matmul(
        jnp.matmul((mu_pris_param - mu_dist_param), invcov_dist_params),
        jnp.transpose((mu_pris_param - mu_dist_param))
    )
    
    """
    vec_means = (mu_pris_param - mu_dist_param).squeeze()
    cc_mtx = (cov_pris_param + cov_dist_param)/2
    L = jax.lax.stop_gradient(jnp.linalg.cholesky(cc_mtx))
    v = jsp.linalg.solve(L, vec_means.squeeze())
    val = jnp.inner(v, v)
    """
    quality = jnp.sqrt(val).squeeze()
    return quality


def _scale_features(features):
    lower_bound = -1
    upper_bound = 1
    feature_ranges = jnp.array([
        [0.338, 10], [0.017204, 0.806612], [0.236, 1.642],
        [-0.123884, 0.20293], [0.000155, 0.712298], [0.001122, 0.470257],
        [0.244, 1.641], [-0.123586, 0.179083], [0.000152, 0.710456],
        [0.000975, 0.470984], [0.249, 1.555], [-0.135687, 0.100858],
        [0.000174, 0.684173], [0.000913, 0.534174], [0.258, 1.561],
        [-0.143408, 0.100486], [0.000179, 0.685696], [0.000888, 0.536508],
        [0.471, 3.264], [0.012809, 0.703171], [0.218, 1.046],
        [-0.094876, 0.187459], [1.5e-005, 0.442057], [0.001272, 0.40803],
        [0.222, 1.042], [-0.115772, 0.162604], [1.6e-005, 0.444362],
        [0.001374, 0.40243], [0.227, 0.996], [-0.117188, 0.09832299999999999],
        [3e-005, 0.531903], [0.001122, 0.369589], [0.228, 0.99], [-0.12243, 0.098658],
        [2.8e-005, 0.530092], [0.001118, 0.370399]])

    scaled_features = lower_bound + (upper_bound - lower_bound) * (features - feature_ranges[:, 0]) / (
            feature_ranges[:, 1] - feature_ranges[:, 0])

    return scaled_features


def niqe(img: jnp.ndarray, kernel_size, kernel_sigma, data_range = 2.):
    loaded = np.load('iqa_funs/niqe_pris_params.npz')
    mu_pris_param = jnp.array(loaded['mu_pris_param'])
    cov_pris_param = jnp.array(loaded['cov_pris_param'])
    gaussian_window = _gaussian_filter(kernel_size, kernel_sigma)

    img = img / float(data_range) * 255

    calc_func = partial(
        _calculate_niqe, mu_pris_param=mu_pris_param, cov_pris_param=cov_pris_param, gaussian_window=gaussian_window)
    quality = calc_func(img)
    return quality


def _gaussian_filter(kernel_size: int = 7, sigma: float = 7 / 6) -> jnp.ndarray:
    """
    Create a 2D Gaussian filter.
    Args:
        kernel_size: Size of the Gaussian kernel.
        sigma: Standard deviation of the Gaussian kernel.
    Returns:
        2D Gaussian kernel.
    """
    x = jnp.arange(-kernel_size//2+1, kernel_size//2+1)
    window = jsp.stats.norm.pdf(x, scale=sigma) * jsp.stats.norm.pdf(x[:, None], scale=sigma)
    window = window / jnp.sum(window)
    return window


# You might need to adapt the `_Loss` class to your requirements, using functions instead of classes.
@jax.jit
def niqe_loss(x: jnp.ndarray) -> jnp.ndarray:
    kernel_size = 7
    kernel_sigma = 7. / 6
    data_range = 2.
    x = x.squeeze()
    return niqe(x, kernel_size, kernel_sigma, data_range)
