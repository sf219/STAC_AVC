import numpy as np
from STAC_AVC.hessian_compute.toolkit_iagft import get_transform_basis, proy_Q_table
from STAC_AVC.AVC_transform import nint_AVC_transform


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
        self.quant_scal = 1
        self.N = 4

    def set_basis(self):
        self.centroids = self.q_ops_obj.get_centroids()
        self.get_transform_basis()
        self.proy_Q_table()

    def get_transform_basis(self):
        self.eigvecs_list, self.eigvals_list, self.q_mtx = get_transform_basis(self.centroids)

    def proy_Q_table(self):
        self.Q = []
        self.chroma_Q = []
        for i in range(self.centroids.shape[0]):
            U = self.eigvecs_list[i]
            produ_Q = proy_Q_table(self.base_Q, U)
            produ_C = proy_Q_table(self.base_C, U)
            table_Q = self.quant_scal*produ_Q
            table_C = self.quant_scal*produ_C
            self.Q.append(table_Q)
            self.chroma_Q.append(table_C)

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
        qstep = (2**((QP-12)//3)) * 0.5
        term = np.round(W/(qstep*self.Q[ind])).astype(int)
        Z = term
        return Z
    
    def inv_quantization(self, Z, QP, ind=None):
        # q is qbits
        qstep = (2**((QP-12)//3)) * 0.5
        Wout_1 = Z * qstep * self.Q[ind]
        return Wout_1

    def get_weights(self, ind=None):
        return self.centroids[ind, :]
    