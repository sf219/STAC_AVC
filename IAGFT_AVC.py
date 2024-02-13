from scipy.stats import mode
from STAC_AVC.IAGFT_transform import base_iagft_transform as perceptual_transform
from STAC_AVC.SAVC import SAVC
from STAC_AVC.utils_avc import enc_cavlc
from STAC_AVC.SAVC import get_predpel_from_mtx, predict_4_blk, predict_16_blk, mode_header_4, mode_header_16, get_num_zeros
import numpy as np
from numba import njit

@njit
def compute_sae(res_block, wh):
    return np.abs(wh*res_block).sum()

@njit
def compute_sse(res_block, wh):
    return np.square(np.sqrt(wh)*res_block).sum()


class IAGFT_AVC(SAVC):

    def __init__(self, nqs, compute_Q_obj, q_ops_obj, flag_sae=False, flag_uniform=True):
        super().__init__(nqs, flag_uniform=flag_uniform, flag_sae=flag_sae)
        self.compute_Q_obj = compute_Q_obj
        self.q_ops_obj = q_ops_obj  
        self.trans = perceptual_transform(compute_Q_obj, q_ops_obj=q_ops_obj, flag_uniform=flag_uniform)
        if self.flag_sae:
            self.distortion_function = lambda x, y: compute_sae(x, y)
        else:
            self.distortion_function = lambda x, y: compute_sse(x, y)

    def set_Q(self, Seq):
        self.h, self.w = Seq.shape        
        img_quan = 2*Seq/255
        self.Qmtx = self.compute_Q_obj.sample_q(Seq)
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

    def final_bits(self, bits):
        return bits + self.overhead_bits

    def set_quantization_parameters(self, ind_quality):
        self.QP = self.qsnu[ind_quality]
        self.lam = 0.85 * ((2**((self.QP-12)//3)) * 0.5)**2 # taken from FHHI
        if self.flag_sae:
            self.lam = np.sqrt(self.lam)

    # this function ensures that the Q-GFT is applied uniformly to blocks of size 8x8
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
        
        # these are the final codewords for the Q-GFT. We will use them to compute the overhead bits
        self.blk_compress = np.zeros(((self.h//(2*self.b_size), self.w//(2*self.b_size))))

    def compress_Q(self):
        self.overhead_bits, _ = self.q_ops_obj.compress_q(self.blk_compress)
        print(' overhead bits: ', self.overhead_bits)

    def encode_i_frame(self, Seq):
        h, w = Seq.shape

        Pred = np.zeros(Seq.shape)
        Res = np.zeros(Seq.shape)
        self.Seq_r = np.zeros((Seq.shape[0], Seq.shape[1]))
        
        self.modes_4 = np.zeros((Seq.shape[0]//self.b_size, Seq.shape[1]//self.b_size), dtype=int)
        self.num_zeros = np.zeros(((h//self.b_size), (w//self.b_size)), dtype=int)

        bits_frame = ''

        for i in range(0, h, self.mb_size):
            for j in range(0, w, self.mb_size):
                idx = i // self.b_size
                jdx = j // self.b_size
                Seq_tmp = Seq[i:i+self.mb_size, j:j+self.mb_size]
                Seq_r1, Pred1, bits_frame1, sae1, Res1, num_zeros_4 = self.intra_4_mb(Seq_tmp, (i, j))
                Seq_r2, Pred2, bits_frame2, sae2, Res2, num_zeros_16 = self.intra_16_mb(Seq_tmp, (i, j))
        
                flag = self.RDO(sae1, sae2, bits_frame1, bits_frame2)

                if flag == 1:
                    Seq_r_tmp = Seq_r1
                    Pred_tmp = Pred1
                    bits_frame += '1' + bits_frame1
                    Res_tmp = Res1
                    self.num_zeros = num_zeros_4
                    self.blk_compress[idx//2:(idx//2)+2, jdx//2:(jdx//2)+2] = self.blk_class_4[idx:idx+4:2, jdx:jdx+4:2]                    
                else:
                    Seq_r_tmp = Seq_r2
                    Pred_tmp = Pred2
                    bits_frame += '0' + bits_frame2
                    Res_tmp = Res2
                    self.modes_4[idx:idx+4, jdx:jdx+4] = 2 # if we choose 16, we set the modes to 2
                    self.num_zeros = num_zeros_16
                    self.blk_compress[idx//2:(idx//2)+2, jdx//2:(jdx//2)+2] = self.blk_class_16[idx, jdx]                    

                self.Seq_r[i:i+self.mb_size, j:j+self.mb_size] = Seq_r_tmp
                Pred[i:i+self.mb_size, j:j+self.mb_size] = Pred_tmp
                Res[i:i+self.mb_size, j:j+self.mb_size] = Res_tmp
        Seq_r = self.Seq_r[0:Seq.shape[0], 0:Seq.shape[1]]
        self.compress_Q()
        return Seq_r, Pred, bits_frame, Res
    
    def intra_4_mb(self, block, block_position):
        total_sae = 0
        bits_frame = ''

        idx = block_position[0]
        jdx = block_position[1]

        Seq_r = np.zeros_like(block)
        Pred = np.zeros_like(block)
        Res = np.zeros_like(block)
        num_zeros_4 = self.num_zeros.copy()

        Seq_r_tmp = self.Seq_r.copy()
        weights = np.ones((self.mb_size, self.mb_size))

        for i in range(0, self.mb_size, self.b_size):
            for j in range(0, self.mb_size, self.b_size):
                blk = block[i:i+self.b_size, j:j+self.b_size]
                cur_pos_mod_1 = (idx + i)//self.b_size
                cur_pos_mod_2 = (jdx + j)//self.b_size

                ind_weights = self.blk_class_4[cur_pos_mod_1, cur_pos_mod_2]
                weights[i:i+4, j:j+4] = self.trans.get_weights(ind_weights)

                predpel = get_predpel_from_mtx(Seq_r_tmp, i+idx, j+jdx)
                wh = weights[i:i+4, j:j+4]

                icp, pred, sae, mode = predict_4_blk(blk, predpel, (idx+i, jdx+j))

                self.modes_4[cur_pos_mod_1, cur_pos_mod_2] = mode
                
                bits_m = mode_header_4(self.modes_4, idx+i, jdx+j)

                icp_r_block, bits_b, num_zeros_4 = self.code_block_4(icp, num_zeros_4, cur_pos_mod_1, cur_pos_mod_2)
                bits_frame += bits_b + bits_m
                total_sae += self.distortion_function(icp_r_block + pred - blk, wh)
                Pred[i:i+self.b_size, j:j+self.b_size] = pred
                Res[i:i+self.b_size, j:j+self.b_size] = icp
                Seq_r[i:i+self.b_size, j:j+self.b_size] = icp_r_block + pred
                Seq_r_tmp[idx+i:idx+i+self.b_size, jdx+j:jdx+j+self.b_size] = icp_r_block + pred
        return Seq_r, Pred, bits_frame, total_sae, Res, num_zeros_4

    def intra_16_mb(self, block, block_position):
        total_sae = 0
        bits_frame = ''

        idx = block_position[0]
        jdx = block_position[1]

        ind_weights = self.blk_class_16[idx//4, jdx//4]
        weights = self.trans.get_weights(ind_weights)

        # stack weights in a 4x4 array
        weights = np.tile(weights, (4, 4))

        num_zeros_16 = self.num_zeros.copy()
        # stack weights in a 4x4 array
        icp, pred, sae, mode = predict_16_blk(block, self.Seq_r, idx, jdx)
        bits_m = mode_header_16(mode, idx, jdx)
        icp_r_block, bits_b, num_zeros_16 = self.code_block_16(icp, num_zeros_16, idx//4, jdx//4)
        bits_frame += bits_b + bits_m
        total_sae += self.distortion_function(icp_r_block + pred - block, weights)
        Pred = pred
        Res = icp
        Seq_r = icp_r_block + pred
        return Seq_r, Pred, bits_frame, total_sae, Res, num_zeros_16

    def code_block_4(self, err, num_zeros_4, ix, jx):
        err_r, bits_tmp, num_zeros_4 = self.transcoding(err, num_zeros_4, (ix, jx))
        bits = bits_tmp
        return err_r, bits, num_zeros_4

    def simple_code_block_16(self, err, num_zeros_16, ix, jx):
        n = self.mb_size
        m = self.mb_size
        b_size = self.b_size
        err_r = np.zeros((n, m))
        bits = ''
        for i in range(0, n, b_size):
            for j in range(0, m, b_size):
                blk = err[i:i+b_size, j:j+b_size]
                idx = ix + i//b_size
                jdx = jx + j//b_size
                err_r[i:i+b_size, j:j+b_size], bits_tmp, num_zeros_16 = self.transcoding(blk, num_zeros_16, (idx, jdx))
                bits += bits_tmp
        return err_r, bits, num_zeros_16

    def transcoding(self, blk, num_zeros, pos):
        QP = self.QP
        idx = pos[0]
        idy = pos[1]
        cp = self.trans.fwd_pass(blk, QP,  ind=self.blk_class_4[idx, idy])
        num_zeros[idx, idy] = np.count_nonzero(cp)
        nL, nU = get_num_zeros(num_zeros, idx, idy)
        bits = enc_cavlc(cp, nL, nU)
        err_r = self.trans.bck_pass(cp, QP, ind=self.blk_class_4[idx, idy])
        return err_r, bits, num_zeros

    def code_block_16(self, err, num_zeros_16, ix, jx):
        QP = self.QP
        n = self.mb_size
        m = self.mb_size
        b_size = self.b_size
        err_r = np.zeros((n, m))
        bits = ''

        cq = np.zeros((n, m), dtype=int)
        c_dc = np.zeros((b_size, b_size))
        for i in range(0, n, b_size):
            for j in range(0, m, b_size):
                class_label = self.blk_class_16[ix + i//b_size, jx + j//b_size]                
                c = self.trans.integer_transform(err[i:i+b_size, j:j+b_size], ind=class_label)
                cq[i:i+b_size, j:j+b_size] = self.trans.quantization(c, QP, ind=class_label)
                c_dc[i//b_size, j//b_size] = c[0, 0]

        #c_dc = c[0::4, 0::4]
        c_dc = self.trans.secondary_integer_transform(c_dc)
        c_dc_q = self.trans.secondary_quantization(c_dc, QP, ind=self.blk_class_16[ix, jx])

        for i in range(0, n, b_size):
            for j in range(0, m, b_size):
                cq_tmp = cq[i:i+b_size, j:j+b_size]
                cq_tmp[0, 0] = c_dc_q[i//b_size, j//b_size]
                num_zeros_16[ix + i//b_size, jx + j//b_size] = np.count_nonzero(cq_tmp)
                idx = ix + i//b_size
                jdx = jx + j//b_size
                nL, nU = get_num_zeros(num_zeros_16, idx, jdx)
                bits += enc_cavlc(cq_tmp, nL, nU)
        
        cq[0::4, 0::4] = self.trans.inverse_secondary_integer_transform(c_dc_q) / 4
        for i in range(0, n, b_size):
            for j in range(0, m, b_size):
                blk = cq[i:i+b_size, j:j+b_size]
                class_label = self.blk_class_16[ix + i//b_size, jx + j//b_size]                
                err_r[i:i+b_size, j:j+b_size] = self.trans.bck_pass(blk, QP, ind=class_label)
        return err_r, bits, num_zeros_16

