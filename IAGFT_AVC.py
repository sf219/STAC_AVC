from scipy.stats import mode
from STAC_AVC.IAGFT_transform import base_iagft_transform as perceptual_transform
from STAC_AVC.SAVC import SAVC
import numpy as np


class IAGFT_AVC(SAVC):

    def __init__(self, nqs, compute_Q_obj, q_ops_obj, flag_uniform=True):
        super().__init__(nqs)
        self.compute_Q_obj = compute_Q_obj
        self.q_ops_obj = q_ops_obj  
        self.trans = perceptual_transform(compute_Q_obj, flag_uniform)

    def set_Q(self, Seq):
        img_quan = 2*Seq/255        
        self.Qval, self.ind_closest, _ = self.compute_Q_obj.sample_q(Seq)
        self.Qmtx = self.compute_Q_obj.sample_q(input)
        self.Qmtx = self.q_ops_obj.normalize_q(self.Qmtx)
        self.q_ops_obj.quantize_q(self.Qmtx, img_quan)
        self.q_ops_obj.choose_ncwd()
        self.overhead_bits = self.q_ops_obj.overhead_bits
        self.centroids = self.q_ops_obj.centroids
        self.ind_closest = self.q_ops_obj.ind_closest
        print(' overhead bits: ', self.overhead_bits, ' centroids: ', self.centroids.shape[0])
        self.ind_closest_420 = self.q_ops_obj.ind_closest_420
        self.Q_quantized = self.q_ops_obj.Q
        self.trans.set_basis()    
        self.compute_class_Q_frame()
        self.compress_Q()

    def final_bits(self, bits):
        return bits + self.overhead_bits

    def compute_class_Q_frame(self):
        tmp_class = self.ind_closest
        tmp_class = tmp_class.astype(int)
        self.blk_compress = np.zeros((tmp_class.shape[0]//2, tmp_class.shape[1]//2))
        self.blk_class_16 = np.zeros_like(tmp_class)
        self.blk_class_4 = np.zeros_like(tmp_class)
        for i in range(0, tmp_class.shape[0], 4):
            for j in range(0, tmp_class.shape[1], 4):
                tmp_all = tmp_class[i:i+4, j:j+4]
                self.blk_class_16[i:i+4, j:j+4] = mode(tmp_all.ravel())[0]
    
        for i in range(0, tmp_class.shape[0], 2):
            for j in range(0, tmp_class.shape[1], 2):
                tmp_all = tmp_class[i:i+2, j:j+2]
                self.blk_class_4[i:i+2, j:j+2] = mode(tmp_all.ravel())[0]

    def compress_Q(self):
        self.overhead_bits = self.q_ops_obj.compress_q(self.blk_compress)
