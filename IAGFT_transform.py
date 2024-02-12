import numpy as np
from utils.utils_lpit import unifgrid, inv_zig_zag
from scipy.linalg import eigh
from scipy.fftpack import dct, idct
from video_codecs.STAC_AVC.AVC_transform import nint_AVC_transform


# we are working with JPEG, so it's safe to set N = 8
N = 8
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
        bas = inv_zig_zag(bas)
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


def proy_Q_table(table, basis):
    Uq = np.abs(D_2d.T @ basis)
    Uq = Uq @ np.linalg.inv(np.diag(np.sum(Uq, axis=0)))
    produ = np.abs(table.ravel('F').T @ Uq)
    produ = produ.reshape((N, N), order='F')
    return produ

class base_iagft_transform(nint_AVC_transform):

    def __init__(self, compute_Q_obj, q_ops_obj, flag_uniform=True):
        super().__init__()
        self.compute_Q_obj = compute_Q_obj
        self.q_ops_obj = q_ops_obj  
        self.flag_uniform = flag_uniform
        if flag_uniform:
            self.base_Q = np.ones((4, 4))
            self.base_C = np.ones((4, 4))
        else:
            self.base_Q = (np.array([[6, 13, 20, 28],
                                    [13, 20, 28, 32],
                                    [20, 28, 32, 37],
                                    [28, 32, 37, 42]])/6)
            self.base_C = (np.array([[6, 13, 20, 28],
                                    [13, 20, 28, 32],
                                    [20, 28, 32, 37],
                                    [28, 32, 37, 42]])/6)
        self.centroids = self.compute_Q_obj.get_centroids(0)
        self.quant_scal = 1
        self.get_transform_basis()
        self.proy_Q_table()

    def set_basis(self):
        self.get_transform_basis()
        self.proy_Q_table()

    def get_transform_basis(self):
        L, _ = unifgrid(self.N)
        self.eigvecs_list = []
        self.eigvals_list = []
        self.q_mtx = []
        centroids = self.q_ops_obj.get_centroids()
        for i in range(centroids.shape[0]):
            q_val = (centroids[i, :])
            eigvecs, eigvals = compute_iagft_basis(np.diag(q_val.ravel('F')), L)
            self.eigvecs_list.append(eigvecs)
            self.q_mtx.append(np.diag(q_val.ravel('F')))
            self.eigvals_list.append(eigvals)

    def proy_Q_table(self):
        self.Q = []
        self.chroma_Q = []
        for j in range(self.nqs):
            qf = self.quant[j]
            Q_inner = []
            chroma_Q_inner = []
            for i in range(self.centroids.shape[0]):
                U = self.eigvecs_list[i]
                produ_Q = proy_Q_table(self.base_Q, U)
                produ_C = proy_Q_table(self.base_C, U)
                table_Q = qf*self.quant_scal*produ_Q
                table_C = qf*self.quant_scal*produ_C
                Q_inner.append(table_Q)
                chroma_Q_inner.append(table_C)
            self.Q.append(Q_inner)
            self.chroma_Q.append(chroma_Q_inner)

    def integer_transform(self, X, ind=None):
        C = self.eigvecs_list[ind]
        Q = self.q_mtx[ind]
        tmp = C.T @ Q @ X.ravel('F')
        tmp = tmp.reshape((self.N, self.N), order='F')
        return tmp
    
    def inv_integer_transform(self, W, ind=None):
        C = self.eigvecs_list[ind]
        tmp = C @ W.ravel('F')
        tmp = tmp.reshape((self.N, self.N), order='F')
        return tmp

    def quantization(self, W, QP, ind=None):    
        # q is qbits
        # Scaling and quantization
        qstep = (2**((QP-12)//3)) * 0.125
        term = np.round(W/(qstep*self.Q[ind])).astype(int)
        Z = term
        return Z
    
    def inv_quantization(self, Z, QP, ind=None):
        # q is qbits
        qstep = (2**((QP-12)//3)) * 0.125
        Wout_1 = Z * qstep * self.Q[ind]
        return Wout_1

    def get_weights(self, ind=None):
        return self.centroids[ind, :]
    