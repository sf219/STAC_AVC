import numpy as np
from scipy.fftpack import dct, idct
from utils.utils_lpit import non_directed_path, unifgrid
from utils.q_utils import decompose_Q
from scipy.fftpack import dct
from scipy.linalg import eigh


def eigen_decomp_grid(Q, L, L_path):
    N = int(np.sqrt(L.shape[0]))
    Q1, Q2 = decompose_Q(Q)
    Q = np.kron(Q1, Q2)
    eigvals, eigvecs = eigh(L, Q, eigvals_only=False)
    eigvals = np.real(eigvals)
    # sort eig_vals in descending order
    eigvecs = np.real(eigvecs)
    #full_eigvecs = np.kron(eigvecs_rows, eigvecs_cols)

    #inds = np.argmax(np.abs(eigvecs.T @ Q @ full_eigvecs), axis=1)
    #eigvecs = full_eigvecs[:, inds]

    match = np.arange(N**2)
    help_basis = np.zeros_like(eigvecs)
    D = dct(np.eye(N), norm='ortho', axis=0).T
    for p in range(0, N**2):
        bas = np.zeros((N**2, N**2))
        bas[p, p] = 1
        bas = np.diag(bas)
        bas = np.reshape(bas, (N, N), order='F')
        eig_ssp = D @ bas @ D.T
        eig_ssp = eig_ssp.ravel('F')
        inds = np.zeros((N**2))
        for j in match:
            inds[j] = np.abs(eigvecs[:, j].T @ Q @ eig_ssp)
        ref = np.argmax(inds)
        help_basis[:, p] = eigvecs[:, ref]
        # remove ref from match
        match = match[match != ref]

    #inds = np.argmax(np.abs(eigvecs.T @ Q @ help_basis), axis=0)
    #eigvecs = eigvecs[:, inds]
    eigvecs = help_basis

    D = np.kron(D, D)

    proy = eigvecs.T @ D
    sign_mtx = np.diag(np.sign(np.diag(proy)))
    eigvecs = eigvecs @ sign_mtx
    eigvecs_Q = np.array(eigvecs)
    return eigvecs_Q, eigvals


def eigen_decomp(Q, L):
    eigvals, eigvecs = eigh(L, Q, eigvals_only=False)
    eigvals = np.real(eigvals)
    # sort eig_vals in descending order
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]    
    eigvecs = np.real(eigvecs)
    D = dct(np.eye(L.shape[0]), norm='ortho', axis=0).T
    proy = eigvecs.T @ D
    sign_mtx = np.diag(np.sign(np.diag(proy)))
    eigvecs = eigvecs @ sign_mtx
    eigvecs_Q = np.array(eigvecs)
    return eigvecs_Q, eigvals

class AVC_transform:

    def __init__(self, flag_uniform=True):
        self.Cf = np.array([[1, 1, 1, 1],
                            [2, 1, -1, -2],
                            [1, -1, -1, 1],
                            [1, -2, 2, -1]])
        
        self.Ci = np.array([[1, 1, 1, 1],
                            [1, 1/2, -1/2, -1],
                            [1, -1, -1, 1],
                            [1/2, -1, 1, -1/2]])
        
        self.sec_Cf = np.array([[1, 1, 1, 1],
                                [1, 1, -1, -1],
                                [1, -1, -1, 1],
                                [1, -1, 1, -1]])
        
        self.sec_Ci = np.array([[1, 1, 1, 1],
                                [1, 1, -1, -1],
                                [1, -1, -1, 1],
                                [1, -1, 1, -1]])
        
        self.SM = np.array([[10, 16, 13],
                            [11, 18, 14],
                            [13, 20, 16],
                            [14, 23, 18],
                            [16, 25, 20],
                            [18, 29, 23]])
        
        self.MF = np.array([[13107, 5243, 8066],
                            [11916, 4660, 7490],
                            [10082, 4194, 6554],
                            [9362, 3647, 5825],
                            [8192, 3355, 5243],
                            [7282, 2893, 4559]])
        
        if flag_uniform:
            self.base_Q = np.ones((4, 4))
        else:
            self.base_Q = np.array([[6, 13, 20, 28],
                                    [13, 20, 28, 32],
                                    [20, 28, 32, 37],
                                    [28, 32, 37, 42]])/16


    def inv_integer_transform(self, W, ind=None):
        Ci = self.Ci
        Y = Ci.T @ W @ Ci
        return Y

    def integer_transform(self, X, ind=None):
        C = self.Cf
        W = (C @ X @ C.T)
        return W

    def secondary_integer_transform(self, X):
        C = self.sec_Cf
        W = (C @ X @ C)/2
        return W

    def inverse_secondary_integer_transform(self, X):
        C = self.sec_Ci
        W = (C @ X @ C)

        return W

    def inv_quantization(self, Z, QP, ind=None):
        # q is qbits
        pres = np.floor(QP/6).astype(int)

        # The scaling factor matrix V depends on the QP and the position of the coefficient.
        SM = self.SM

        x = int(QP % 6)

        # Find delta, lambda, and miu values
        d = SM[x, 0]
        l = SM[x, 1]
        m = SM[x, 2]

        V = np.array([[d, m, d, m],
                    [m, l, m, l],
                    [d, m, d, m],
                    [m, l, m, l]]).astype(int)

        # Find the inverse quantized coefficients
        Wi = Z * V
        Wout_1 = (Wi << pres)
        return Wout_1

    def quantization(self, W, QP, ind=None):
        # q is qbits
        q = 15 + np.floor(QP / 6).astype(int)

        # M is the multiplying factor which is found from QP value
        # MF is the multiplying factor matrix
        # rem(QP,6) alpha   beta    gamma
        #           (a)     (b)      (g)
        # 0         13107   5243    8066
        # 1         11916   4660    7490
        # 2         10082   4194    6554
        # 3         9362    3647    5825
        # 4         8192    3355    5243
        # 5         7282    2893    4559

        MF = self.MF

        x = int(QP % 6)

        a = MF[x, 0]
        b = MF[x, 1]
        g = MF[x, 2]

        M = np.array([[a, g, a, g],
                    [g, b, g, b],
                    [a, g, a, g],
                    [g, b, g, b]])

        # Scaling and quantization
        term = (W * M + 2**(q-1)).astype(int)
        Z = (term >> q)
        return Z
        
    def fwd_pass(self, X, QP, ind=None):
        W = self.integer_transform(X, ind=ind)
        Z = self.quantization(W, QP, ind=ind)
        return Z
    
    def bck_pass(self, Z, QP, ind=None):
        W = self.inv_quantization(Z, QP, ind=ind)
        Y = self.inv_integer_transform(W)
        err_r = (Y + 32).astype(int) >> 6
        return err_r
    
    def secondary_quantization(self, W, QP, ind=None):
        Z = self.quantization(W, QP, ind=ind)
        """
        q = 16 + np.floor(QP / 6).astype(int)
        MF = np.array([[13107, 5243, 8066],
                    [11916, 4660, 7490],
                    [10082, 4194, 6554],
                    [9362, 3647, 5825],
                    [8192, 3355, 5243],
                    [7282, 2893, 4559]])
        x = int(QP % 6)
        a = MF[x, 0]
        # Scaling and quantization
        term = (W * a + 2**(q-1)).astype(int)
        Z = (term >> q)
        """
        return Z

    def get_weights(self, ind=None):
        return np.ones((4, 4))
    

class nint_AVC_transform(AVC_transform):

    def __init__(self, flag_uniform=True):
        super().__init__()

        self.Cf = dct(np.eye(4), norm='ortho', axis=0)
        self.Ci = self.Cf
        if flag_uniform:
            self.base_Q = np.ones((4, 4))
        else:
            self.base_Q = np.array([[6, 13, 20, 28],
                                    [13, 20, 28, 32],
                                    [20, 28, 32, 37],
                                    [28, 32, 37, 42]])/6

    
    def bck_pass(self, Z, QP, ind=None):
        W = self.inv_quantization(Z, QP, ind=ind)
        Y = self.inv_integer_transform(W, ind=ind)
        return Y
    
    def quantization(self, W, QP, ind=None):    
        # q is qbits
        # Scaling and quantization
        qstep = (2**((QP-12)//3)) * 0.125
        term = np.round(W/(qstep*self.base_Q)).astype(int)
        Z = term
        return Z
    
    def inv_quantization(self, Z, QP, ind=None):
        # q is qbits
        qstep = (2**((QP-12)//3)) * 0.125
        Wout_1 = Z * qstep * self.base_Q
        return Wout_1
    
    def secondary_quantization(self, W, QP, ind=None):
        return self.quantization(W, QP, ind)
    

class nint_MSSSIM_transform(nint_AVC_transform):

    def __init__(self, compute_Q_obj, flag_uniform=True):
        super().__init__()
        self.compute_Q_obj = compute_Q_obj
        self.flag_uniform = flag_uniform
        if flag_uniform:
            self.base_Q = np.ones((4, 4))
        else:
            self.base_Q = (np.array([[6, 13, 20, 28],
                                    [13, 20, 28, 32],
                                    [20, 28, 32, 37],
                                    [28, 32, 37, 42]])/6)
        self.centroids = self.compute_Q_obj.get_centroids(0)
        self.get_transform_basis()
        self.proy_Q_table()

    def get_transform_basis(self):
        L, _ = non_directed_path(4)
        self.eigvecs_list_rows = []
        self.eigvecs_list_cols = []
        self.eigvals_list_rows = []
        self.eigvals_list_cols = []
        self.q_cols = []
        self.q_rows = []
        centroids = self.centroids
        for i in range(self.compute_Q_obj.n_cwd):
            q_val = (centroids[i, :])
            Q1, Q2 = decompose_Q(np.diag(q_val.ravel('F')))
            self.q_cols.append(np.diag(Q2))
            self.q_rows.append(np.diag(Q1))
            eigevecs_cols, eigvals_cols = eigen_decomp(Q2, L)
            eigevecs_rows, eigvals_rows = eigen_decomp(Q1, L)
            self.eigvecs_list_cols.append(eigevecs_cols)
            self.eigvecs_list_rows.append(eigevecs_rows)
            self.eigvals_list_cols.append(eigvals_cols)
            self.eigvals_list_rows.append(eigvals_rows)

    def proy_Q_table(self):
        self.Q = []
        D = dct(np.eye(4), norm='ortho', axis=0).T
        for i in range(self.compute_Q_obj.n_cwd):
            tmp_Q = (self.base_Q) 
            #if self.uniform:
            #    Q_inner.append(tmp_Q)
            #    continue     
            U_rows = self.eigvecs_list_rows[i] 
            U_cols = self.eigvecs_list_cols[i]

            Uq_rows = np.abs(D.T @ U_rows)
            Uq_cols = np.abs(D.T @ U_cols)

            Uq_rows = Uq_rows/np.sum(Uq_rows, axis=0)
            Uq_cols = Uq_cols/np.sum(Uq_cols, axis=0)
            #breakpoint()
            produ = Uq_cols @ tmp_Q @ Uq_rows.T
            self.Q.append(produ)

    def integer_transform(self, X, ind=None):
        C_cols = self.eigvecs_list_cols[ind]
        C_rows = self.eigvecs_list_rows[ind]
        Q_cols = np.diag(self.q_cols[ind])
        Q_rows = np.diag(self.q_rows[ind])
        return C_cols.T @ Q_cols @ X @ Q_rows @ C_rows
    
    def inv_integer_transform(self, W, ind=None):
        C_cols = self.eigvecs_list_cols[ind]
        C_rows = self.eigvecs_list_rows[ind]
        return C_cols @ W @  C_rows.T

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
    


class nint_grid_MSSSIM_transform(nint_AVC_transform):

    def __init__(self, compute_Q_obj, flag_uniform=True):
        super().__init__()
        self.compute_Q_obj = compute_Q_obj
        self.flag_uniform = flag_uniform
        self.N = 4
        if flag_uniform:
            self.base_Q = np.ones((4, 4))
        else:
            self.base_Q = (np.array([[6, 13, 20, 28],
                                    [13, 20, 28, 32],
                                    [20, 28, 32, 37],
                                    [28, 32, 37, 42]])/6)
        self.centroids = self.compute_Q_obj.get_centroids(0)
        self.get_transform_basis()
        self.proy_Q_table()

    def get_transform_basis(self):
        L, _ = non_directed_path(4)
        L_grid, _ = unifgrid(4)
        self.eigvecs_list = []
        self.eigvals_list= []
        self.q_mtx = []
        centroids = self.centroids
        for i in range(self.compute_Q_obj.n_cwd):
            q_val = (centroids[i, :])
            self.q_mtx.append(np.diag(q_val.ravel('F')))
            eigvecs, eigvals = eigen_decomp_grid(np.diag(q_val.ravel('F')), L_grid, L)

            self.eigvecs_list.append(eigvecs)
            self.eigvals_list.append(eigvals)


    def proy_Q_table(self):
        self.Q = []
        D = dct(np.eye(4), norm='ortho', axis=0).T
        D = np.kron(D, D)
        for i in range(self.compute_Q_obj.n_cwd):
            tmp_Q = (self.base_Q)
            #if self.uniform:
            #    Q_inner.append(tmp_Q)
            #    continue

            # if self.uniform:
            #     Q_inner.append(tmp_Q)
            #     continue     
            U = self.eigvecs_list[i]

            Uq = np.abs(D.T @ U)
            #Uq = np.abs(D.T @ U)
            Uq = Uq @ np.linalg.inv(np.diag(np.sum(Uq, axis=0)))
            #produ = Uq @ tmp_Q @ Uq.T
            produ = np.abs(tmp_Q.ravel('F').T @ Uq)
            produ = produ.reshape((self.N, self.N), order='F')
            self.Q.append(produ)

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
    

class int_grid_MSSSIM_transform(nint_AVC_transform):

    def __init__(self, compute_Q_obj, flag_uniform=True):
        super().__init__()
        self.compute_Q_obj = compute_Q_obj
        self.flag_uniform = flag_uniform
        self.N = 4
        if flag_uniform:
            self.base_Q = np.ones((4, 4))
        else:
            self.base_Q = (np.array([[6, 13, 20, 28],
                                    [13, 20, 28, 32],
                                    [20, 28, 32, 37],
                                    [28, 32, 37, 42]])/6)
        self.centroids = self.compute_Q_obj.get_centroids(0)
        self.get_transform_basis()
        self.proy_Q_table()

    def get_transform_basis(self):
        L, _ = non_directed_path(4)
        L_grid, _ = unifgrid(4)
        self.eigvecs_fwd_list = []
        self.eigvecs_bck_list = []
        self.eigvals_list= []
        self.q_mtx = []
        self.table_fwd = []
        self.table_bck = []
        centroids = self.centroids
        for i in range(self.compute_Q_obj.n_cwd):
            q_val = (centroids[i, :])
            self.q_mtx.append(np.diag(q_val.ravel('F')))
            eigvecs, eigvals = eigen_decomp_grid(np.diag(q_val.ravel('F')), L_grid, L)

            fwd_t = eigvecs.T @ np.diag(q_val.ravel('F'))
            bck_t = eigvecs

            scale_1 = np.max(np.abs(fwd_t))/4
            scale_2 = np.max(np.abs(bck_t))/4

            fwd_t = np.round(fwd_t / scale_1)
            bck_t = np.round(bck_t / scale_2)

            self.table_fwd.append(scale_1)
            self.table_bck.append(scale_2)

            self.eigvecs_fwd_list.append(fwd_t)
            self.eigvecs_bck_list.append(bck_t)
            self.eigvals_list.append(eigvals)


    def proy_Q_table(self):
        self.Q = []
        D = dct(np.eye(4), norm='ortho', axis=0).T
        D = np.kron(D, D)
        for i in range(self.compute_Q_obj.n_cwd):
            tmp_Q = (self.base_Q)
            #if self.uniform:
            #    Q_inner.append(tmp_Q)
            #    continue

            # if self.uniform:
            #     Q_inner.append(tmp_Q)
            #     continue     
            U = self.eigvecs_fwd_list[i]

            Uq = np.abs(D.T @ U)
            #Uq = np.abs(D.T @ U)
            Uq = Uq @ np.linalg.inv(np.diag(np.sum(Uq, axis=0)))
            #produ = Uq @ tmp_Q @ Uq.T
            produ = np.abs(tmp_Q.ravel('F').T @ Uq)
            produ = produ.reshape((self.N, self.N), order='F')
            self.Q.append(tmp_Q)

    def integer_transform(self, X, ind=None):
        C = self.eigvecs_fwd_list[ind]
        tmp = self.table_fwd[ind] * C @ X.ravel('F')
        tmp = tmp.reshape((self.N, self.N), order='F')
        return tmp
    
    def inv_integer_transform(self, W, ind=None):
        C = self.eigvecs_bck_list[ind]
        tmp = self.table_bck[ind] * C @ W.ravel('F')
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