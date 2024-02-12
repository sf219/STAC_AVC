import numpy as np
from scipy.fftpack import dct
from STAC_AVC.utils_avc import dct_2d

# This file has an implementation of both integer and non-integer transforms for AVC.
# There's almost no difference in performance between both.


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
        return Z

    def get_weights(self, ind=None):
        return np.ones((4, 4))
    

class nint_AVC_transform(AVC_transform):

    def __init__(self, flag_uniform=True):
        super().__init__()

        self.Cf = dct_2d(np.eye(4), norm='ortho', axis=0)
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
    

