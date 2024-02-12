from scipy.stats import mode
from video_codecs.STAC_AVC.AVC_transform import base_iagft_transform as perceptual_transform
from video_codecs.STAC_AVC.SAVC import SAVC


class AVC_MSSIM(SAVC):

    def __init__(self, nqs, compute_Q_obj, flag_uniform=True):
        super().__init__(nqs)
        self.compute_Q_obj = compute_Q_obj
        self.trans = perceptual_transform(compute_Q_obj, flag_uniform)

    def set_Q(self, Seq):
        self.Qval, self.ind_closest, _ = self.compute_Q_obj.sample_q(Seq)
        self.compute_class_Q_frame()

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
        self.overhead_bits = self.compute_Q_obj.compress_Q(self.blk_compress)
