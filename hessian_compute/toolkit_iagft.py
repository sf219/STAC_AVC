import numpy as np
from scipy.fftpack import dct
from STAC_AVC.utils_lpit import unifgrid, inv_zig_zag
from scipy.linalg import eigh


N = 4
D_1d = dct(np.eye(N), norm='ortho', axis=0).T
D_2d = np.kron(D_1d, D_1d)


def fix_sign(basis):
    proy = basis.T @ D_2d
    sign_mtx = np.diag(np.sign(np.diag(proy)))
    basis = basis @ sign_mtx
    return basis


def gevd(L, Q):
    eigvals, eigvecs = eigh(L, Q, eigvals_only=False)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    return eigvals, eigvecs


def sort_inv_zz_dct_basis():
    help_mid_basis = np.zeros((N*N, N*N))
    for p in range(0, N**2):
        bas = np.zeros((N**2, N**2))
        bas[p, p] = 1
        bas = np.diag(bas)
        bas = np.reshape(bas, (N, N), order='F')
        bas = inv_zig_zag(bas, N=N)
        eig_ssp = D_1d @ bas @ D_1d.T
        eig_ssp = eig_ssp.ravel('F')
        help_mid_basis[:, p] = eig_ssp
    return help_mid_basis


def find_matches(inner_prod):
    match = np.ones((N**2))*(N**2 + 1)
    for p in range(0, N**2):
        vector = np.abs(inner_prod[:, p])
        pos = np.argsort(vector)[::-1]
        pos_max = 0
        match_tmp = pos[pos_max] 
        while match_tmp in match:
            pos_max += 1
            match_tmp = pos[pos_max]
        match[p] = match_tmp
    return match


def compute_iagft_basis(Q, L):
    eigvals, eigvecs = gevd(L, Q)
    help_mid_basis = sort_inv_zz_dct_basis()
    inner_prod = eigvecs.T @ Q @ help_mid_basis  
    match = find_matches(inner_prod)
    eigvecs = eigvecs[:, match.astype(np.int32)]
    eigvecs = fix_sign(eigvecs)
    return eigvecs, eigvals


def get_transform_basis(centroids):
    L, _ = unifgrid(N)
    n_cwd = centroids.shape[0]
    eigvecs_list = []
    eigvals_list = []
    q_mtx = []
    for i in range(n_cwd):
        q_val = (centroids[i, :])
        eigvecs, eigvals = compute_iagft_basis(np.diag(q_val.ravel('F')), L)
        eigvecs_list.append(eigvecs)
        q_mtx.append(np.diag(q_val.ravel('F')))
        eigvals_list.append(eigvals)
    return eigvecs_list, eigvals_list, q_mtx


def proy_Q_table(table, basis):
    Uq = np.abs(D_2d.T @ basis)
    Uq = Uq @ np.linalg.inv(np.diag(np.sum(Uq, axis=0)))
    produ = np.abs(table.ravel('F').T @ Uq)
    produ = produ.reshape((N, N), order='F')
    return produ
