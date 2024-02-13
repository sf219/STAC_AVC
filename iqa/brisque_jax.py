import jax.numpy as jnp
import jax.scipy as jsp
from jax.scipy.special import gammaln
import jax.image
import jax
import numpy as np

def gaussian_filter(kernel_size: int = 7, sigma: float = 7 / 6) -> jnp.ndarray:
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
    window = window.reshape((1, 1, window.shape[0], window.shape[1]))
    return window


def brisque(x: jnp.ndarray, kernel_size: int = 7, kernel_sigma: float = 7 / 6,
            data_range: float = 2.) -> jnp.ndarray:
    """
    Compute BRISQUE quality score for a batch of images.
    Args:
        x: Batch of images with shape (batch_size, channels, height, width).
        kernel_size: Size of the Gaussian kernel.
        kernel_sigma: Standard deviation of the Gaussian kernel.
        data_range: Value range of input images (usually 1.0 or 255).
        reduction: Reduction over samples in batch: "mean"|"sum"|"none"
    Returns:
        BRISQUE score for each image in batch.
    """
    #x = x.astype(jnp.float32).reshape((1, 1, x.shape[0], x.shape[1]))
    x = x / float(data_range) * 255 

    features = []
    num_of_scales = 2

    for _ in range(num_of_scales):
        features.append(_natural_scene_statistics(x, kernel_size, kernel_sigma))
        # convert x to PIL image
        x = jax.image.resize(x, (1, 1, x.shape[2] // 2, x.shape[3] // 2), method='cubic')
        # reduce size of x by half

    features = jnp.concatenate(features, axis=-1)
    sc_features = _scale_features(features)
    score = _score_svr(sc_features)
    return jnp.mean(score)


# You might need to adapt the `_Loss` class to your requirements, using functions instead of classes.
@jax.jit
def brisque_loss(x: jnp.ndarray) -> jnp.ndarray:
    kernel_size = 7
    kernel_sigma = 7. / 6
    data_range = 2.
    return brisque(x, kernel_size, kernel_sigma, data_range)


def _ggd_parameters(x):
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


def _aggd_parameters(x):
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


def _natural_scene_statistics(luma, kernel_size=7, sigma=7. / 6):
    window = gaussian_filter(kernel_size, sigma)
    C = 1

    win_size = kernel_size 
    pad = (win_size-1)//2
    lf = jnp.pad(luma.squeeze(), pad, mode='symmetric')
    lf = lf.reshape((1, 1, lf.shape[0], lf.shape[1]))
    
    mu = jsp.signal.convolve(lf, window, mode='valid')
    lf_tmp = (luma.squeeze() - mu.squeeze())**2
    lf_tmp = jnp.pad(lf_tmp, pad, mode='symmetric')
    lf_tmp = lf_tmp.reshape((1, 1, lf_tmp.shape[0], lf_tmp.shape[1]))
    var = jsp.signal.convolve(lf_tmp, window, mode='valid')
    std = jnp.sqrt(var + 0.25)

    luma_nrmlzd = (luma - mu) / (std + 0.5)

    """
    if jnp.isnan(luma_nrmlzd).any():
        print('nan')
        luma_nrmlzd = (luma - mu) / (std + C)
    """

    alpha, sigma = _ggd_parameters(luma_nrmlzd)
    features = [alpha, jnp.square(sigma)]

    shifts = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    
    for shift in shifts:
        shifted_luma_nrmlzd = jnp.roll(luma_nrmlzd, shift, axis=(-2, -1))
        alpha, sigma_l, sigma_r = _aggd_parameters(luma_nrmlzd * shifted_luma_nrmlzd)
        eta = (sigma_r - sigma_l) * jnp.exp(
            gammaln(2. / alpha) - (gammaln(1. / alpha) + gammaln(3. / alpha)) / 2)
        features.extend((alpha, eta, sigma_l**2, sigma_r**2))
    return jnp.stack(features, axis=-1)

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


def _score_svr(features):
    sv_coef = np.load('STAC_AVC/iqa/data_brisque.npz')['sv_coef']
    sv = np.load('STAC_AVC/iqa/data_brisque.npz')['sv']
    # gamma and rho are SVM model parameters taken from official implementation of BRISQUE on MATLAB
    # Source: https://live.ece.utexas.edu/research/Quality/index_algorithms.htm
    gamma = 0.05
    rho = -153.591
    sv_coef = jnp.array(sv_coef).squeeze()
    sv = jnp.array(sv).squeeze()
    kernel_features = _rbf_kernel(features=features, sv=sv.T, gamma=gamma)
    score = kernel_features @ sv_coef
    return score - rho


def _rbf_kernel(features, sv, gamma=0.05):
    dist = jnp.sum(jnp.square(features[:, :, None] - sv[None]), axis=1).squeeze()
    return jnp.exp(- dist * gamma)


def _score_svr_nored(features):
    #url = 'https://github.com/photosynthesis-team/piq/' \
          #'releases/download/v0.4.0/brisque_svm_weights.pt'
    #sv_coef, sv = load_url(url, map_location='cpu')
    sv_coef = np.load('data_brisque.npz')['sv_coef']
    sv = np.load('data_brisque.npz')['sv']
    # gamma and rho are SVM model parameters taken from official implementation of BRISQUE on MATLAB
    # Source: https://live.ece.utexas.edu/research/Quality/index_algorithms.htm
    gamma = 0.05
    rho = -153.591
    sv_coef = jnp.array(sv_coef).squeeze()
    sv = jnp.array(sv).squeeze()
    kernel_features = _rbf_kernel(features=features, sv=sv.T, gamma=gamma)
    score = kernel_features * sv_coef
    return score - rho

def hd_brisque(x: jnp.ndarray, kernel_size: int = 7, kernel_sigma: float = 7 / 6,
            data_range: float = 2., reduction: str = 'mean') -> jnp.ndarray:
    """
    Compute BRISQUE quality score for a batch of images.
    Args:
        x: Batch of images with shape (batch_size, channels, height, width).
        kernel_size: Size of the Gaussian kernel.
        kernel_sigma: Standard deviation of the Gaussian kernel.
        data_range: Value range of input images (usually 1.0 or 255).
        reduction: Reduction over samples in batch: "mean"|"sum"|"none"
    Returns:
        BRISQUE score for each image in batch.
    """
    x = x.astype(jnp.float32).reshape((1, 1, x.shape[0], x.shape[1]))
    x = x / float(data_range) * 255

    features = []
    num_of_scales = 2
    for _ in range(num_of_scales):
        features.append(_natural_scene_statistics(x, kernel_size, kernel_sigma))
        # convert x to PIL image
        x = jax.image.resize(x, (1, 1, x.shape[2] // 2, x.shape[3] // 2), method='cubic')
        # reduce size of x by half

    features = jnp.concatenate(features, axis=-1)
    features = _scale_features(features)
    score = _score_svr_nored(features)
    return score