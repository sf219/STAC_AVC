import numpy as np
from video_codecs.utils_avc import enc_cavlc
from utils.bits_class import Bits
from video_codecs.AVC_transform import AVC_transform as transform
from video_codecs.AVC_transform import int_grid_MSSSIM_transform as perceptual_transform
from scipy.stats import mode

def compute_sae(res_block, weights=1):
    return np.sum(np.square(np.sqrt(weights)*res_block))


def most_probable_mode(modes, i, j):
    if (i > 0 and j > 0):
        a = modes[i-1, j]
        b = modes[i, j-1]
        prev = min(a, b)
    else:
        return False, 0 # it doesn't really matter
    return prev == modes[i, j], prev


def enc_golomb(symbol, sign):
    bits = ''

    # If signed_symbol flag is 1
    if sign == 1:
        if symbol == 0:
            pass
        elif symbol > 0:
            symbol = 2 * symbol - 1
        else:
            symbol = (-2) * symbol

    # If unsigned integers are used
    else:
        pass

    # Here code_num = symbol
    # M is prefix, info is suffix
    M = int(np.floor(np.log2(symbol + 1)))
    info = bin(symbol + 1 - 2 ** M)[2:].zfill(M)

    for j in range(M):
        bits += '0'
    bits += '1'
    bits += info

    return bits


class SAVC():

    def __init__(self, nqs, flag_uniform=True):
        self.qsnu = np.linspace(5.5*nqs, 7*nqs, nqs)
        self.mb_size = 16
        self.b_size = 4
        self.trans = transform(flag_uniform)

    def set_quantization_parameters(self, ind_quality):
        self.QP = self.qsnu[ind_quality]
        self.lam = 0.85 * (2 ** ((self.QP-12)/3)) #taken from FHHI

    def RDO(self, sae1, sae2, bits1, bits2):
        term_1 = sae1 + self.lam*len(bits1)
        term_2 = sae2 + self.lam*len(bits2)

        #print('********************************************')
        #print('RD. Intra 4: ', term_1, ' Intra 16: ', term_2)
        #print('Bits. Intra 4: ', len(bits1), ' Intra 16: ', len(bits2))
        #print('SAE. Intra 4: ', sae1, ' Intra 16: ', sae2)

        if term_1 < term_2:
            return 1
        else:
            return 2

    def compress(self, img, ind_quality=0):
        self.set_quantization_parameters(ind_quality)
        Y, bits, res = self.intra_encode_frame(img)
        bits_ob = Bits()
        bits_ob.bits_over = bits
        Y = np.clip(Y, 0, 255)
        return res, Y, bits_ob
    
    def intra_encode_frame(self, im):
        # Initialize variables
        bitstream = ''
        # Load image sequence
        Seq = np.double(im)
        # Encode I frame
        Seq_r, Pred, bits_frame, Res = self.encode_i_frame(Seq)
        bitstream += bits_frame
        # Calculate number of bits
        num_bits = len(bitstream)
        num_bits += self.overhead_bits
        return Seq_r, num_bits, Res

    def set_Q(self, img):
        self.compute_class_Q_frame(img)

    def compute_class_Q_frame(self, Seq):
        h, w = Seq.shape
        self.blk_class_4 = np.ones((h//self.b_size, w//self.b_size))
        self.blk_class_16 = np.ones_like(self.blk_class_4)
        self.blk_compress = np.zeros(((h//(2*self.b_size), w//(2*self.b_size))))

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
                    self.modes_4[idx:idx+4, jdx:jdx+4] = 2
                    self.num_zeros = num_zeros_16
                    self.blk_compress[idx//2:(idx//2)+2, jdx//2:(jdx//2)+2] = self.blk_class_16[idx, jdx]

                self.Seq_r[i:i+self.mb_size, j:j+self.mb_size] = Seq_r_tmp
                Pred[i:i+self.mb_size, j:j+self.mb_size] = Pred_tmp
                Res[i:i+self.mb_size, j:j+self.mb_size] = Res_tmp
        self.compress_Q()
        Seq_r = self.Seq_r[0:Seq.shape[0], 0:Seq.shape[1]]
        return Seq_r, Pred, bits_frame, Res
    
    def compress_Q(self):
        self.overhead_bits = 0

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

        # decisions are made blockwise.
        weights = np.ones((self.mb_size, self.mb_size))
        for i in range(0, self.mb_size, self.b_size):
            for j in range(0, self.mb_size, self.b_size):
                cur_pos_mod_1 = (idx + i)//self.b_size
                cur_pos_mod_2 = (jdx + j)//self.b_size
                ind_weights = self.blk_class_4[cur_pos_mod_1, cur_pos_mod_2]
                weights[i:i+4, j:j+4] = self.trans.get_weights(ind_weights)
    

        for i in range(0, self.mb_size, self.b_size):
            for j in range(0, self.mb_size, self.b_size):
                blk = block[i:i+self.b_size, j:j+self.b_size]
                cur_pos_mod_1 = (idx + i)//self.b_size
                cur_pos_mod_2 = (jdx + j)//self.b_size

                predpel = get_predpel_from_mtx(Seq_r_tmp, i+idx, j+jdx)
                ind_weights = self.blk_class_4[cur_pos_mod_1, cur_pos_mod_2]

                wh = weights[i:i+4, j:j+4]
                icp, pred, sae, mode = predict_4_blk(blk, predpel, (idx+i, jdx+j), wh)

                self.modes_4[cur_pos_mod_1, cur_pos_mod_2] = mode
                
                bits_m = mode_header_4(self.modes_4, idx+i, jdx+j)

                icp_r_block, bits_b, num_zeros_4 = self.code_block_4(icp, num_zeros_4, cur_pos_mod_1, cur_pos_mod_2)
                bits_frame += bits_b + bits_m
                total_sae += compute_sae(icp_r_block + pred - blk, wh)
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

        num_zeros_16 = self.num_zeros.copy()

        ind_weights = self.blk_class_16[idx//4, jdx//4]
        weights = self.trans.get_weights(ind_weights)

        # stack weights in a 4x4 array
        weights = np.tile(weights, (4, 4))
        icp, pred, sae, mode = predict_16_blk(block, self.Seq_r, idx, jdx, weights)
        bits_m = mode_header_16(mode, idx, jdx)
        icp_r_block, bits_b, num_zeros_16 = self.code_block_16(icp, num_zeros_16, idx//4, jdx//4)
        bits_frame += bits_b + bits_m
        total_sae += compute_sae(icp_r_block + pred - block, weights)
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
        cp = self.trans.fwd_pass(blk, QP, ind=self.blk_class_4[idx, idy])
        num_zeros[idx, idy] = np.sum(np.abs(cp) != 0).astype(int)
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

        cq = np.zeros((n, m)).astype(int)
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
                num_zeros_16[ix + i//b_size, jx + j//b_size] = np.sum(np.abs(cq_tmp) != 0).astype(int)
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


def mode_header_4(modes_4, posx, posy):
    b_size = 4
    cur_pos_mod_1 = (posx)//b_size
    cur_pos_mod_2 = (posy)//b_size
    mode = modes_4[cur_pos_mod_1, cur_pos_mod_2]
    flag_most_prob, most_prob = most_probable_mode(modes_4, cur_pos_mod_1, cur_pos_mod_2)
    if flag_most_prob:
        bits_m = '1'
    else:
        if ((posx) != 0 and (posy) != 0):
            bits_m = '0'
            if mode < most_prob:
                bits_m += enc_golomb(mode, 0)
            else:
                bits_m += enc_golomb(mode-1, 0)
        elif ((posx) == 0 and (posy) == 0):
            bits_m = ''     
        else:
            bits_m = '1'    # because we have two options for the vertical/horizontal blocks
    return bits_m


def mode_header_16(mode, idx, jdx):
    if (idx != 0 and jdx != 0):
        bits_m = enc_golomb(mode, 0)
    elif (idx == 0 and jdx == 0):
        bits_m = ''
    else:
        bits_m = '0'
    return bits_m


def get_num_zeros(num_zeros_mtx, ix, jx):
    if ix == 0 and jx == 0:
        nL = nU = 14
    elif ix == 0:
        nL = nU = num_zeros_mtx[ix, jx - 1]
    elif jx == 0:
        nU = nL = num_zeros_mtx[ix - 1, jx]
    else:
        nL = num_zeros_mtx[ix, jx - 1]
        nU = num_zeros_mtx[ix - 1, jx]
    return nL, nU


def predict_4_blk(blk, predpel, pos, weights):
    xpos = pos[0]
    ypos = pos[1]

    if (xpos) == 0 and (ypos) == 0:
        mode = 9
        icp, pred, sae = no_pred_blk(blk)
    elif (xpos) == 0:
        mode = 1
        icp_1, pred_1, sae_1 = pred_horz_4_blk(blk, predpel)
        icp_2, pred_2, sae_2 = pred_dc_4_blk(blk, 2*predpel)
        if sae_1 < sae_2:
            icp = icp_1
            pred = pred_1
            sae = sae_1
        else:
            icp = icp_2
            pred = pred_2
            sae = sae_2
            mode = 2
    elif (ypos) == 0:
        mode = 0
        icp_1, pred_1, sae_1 = pred_vert_4_blk(blk, predpel)
        icp_2, pred_2, sae_2 = pred_dc_4_blk(blk, 2*predpel)
        if sae_1 < sae_2:
            icp = icp_1
            pred = pred_1
            sae = sae_1
        else:
            icp = icp_2
            pred = pred_2
            sae = sae_2
            mode = 2
    else:
        icp, pred, sae, mode = mode_select_4_blk(blk, predpel, weights)
    return icp, pred, sae, mode

# X A B C D E F G H 
# I a b c d
# J e f g h
# K i j k l
# L m n o p

def get_predpel_from_mtx(Seq_r, i, j):
    pred_pel = np.zeros(13, dtype=int)
    if i > 0 and j > 0:
        pred_pel[0] = Seq_r[i-1, j-1]  # X
    if i > 0:
        pred_pel[1] = Seq_r[i-1, j]    # A
        pred_pel[2] = Seq_r[i-1, j+1]  # B
        pred_pel[3] = Seq_r[i-1, j+2]  # C
        pred_pel[4] = Seq_r[i-1, j+3]  # D
        pred_pel[5] = Seq_r[i-1, j+3]  # E
        pred_pel[6] = Seq_r[i-1, j+3]  # F
        pred_pel[7] = Seq_r[i-1, j+3]  # G
        pred_pel[8] = Seq_r[i-1, j+3]  # H
    if j > 0:
        pred_pel[9] = Seq_r[i, j-1]    # I
        pred_pel[10] = Seq_r[i+1, j-1] # J
        pred_pel[11] = Seq_r[i+2, j-1] # K
        pred_pel[12] = Seq_r[i+3, j-1] # L
    return pred_pel


def pred_horz_4_blk(blk, predpel):
    P_I = predpel[9:]
    pred = np.zeros((4, 4), dtype=int)
    for i in range(4):
        pred[i][0] = pred[i][1] = pred[i][2] = pred[i][3] = P_I[i]
    icp = blk - pred
    sae = compute_sae(icp)
    return icp, pred, sae


def pred_vert_4_blk(blk, predpel):
    selection = predpel[1:5]
    tmp = np.reshape(selection, (1, len(selection)))
    pred = np.tile(tmp, (4, 1))
    icp = blk - pred
    sae = compute_sae(icp)
    return icp, pred, sae


def pred_dc_4_blk(blk, predpel):
    pred = (np.sum(predpel[1:5]) + np.sum(predpel[9:]) + 4).astype(int) >> 3
    icp = blk - pred
    sae = compute_sae(icp)
    return icp, pred, sae


def pred_ddl_4_blk(blk, predpel):
    P_A, P_B, P_C, P_D, P_E, P_F, P_G, P_H = predpel[1:9]
    pred = np.zeros((4, 4), dtype=int)
    pred[0][0] = (P_A + P_C + 2 * P_B + 2) >> 2
    pred[0][1] = pred[1][0] = (P_B + P_D + 2 * P_C + 2) >> 2
    pred[0][2] = pred[1][1] = pred[2][0] = (P_C + P_E + 2 * P_D + 2) >> 2
    pred[0][3] = pred[1][2] = pred[2][1] = pred[3][0] = (P_D + P_F + 2 * P_E + 2) >> 2
    pred[1][3] = pred[2][2] = pred[3][1] = (P_E + P_G + 2 * P_F + 2) >> 2
    pred[2][3] = pred[3][2] = (P_F + P_H + 2 * P_G + 2) >> 2
    pred[3][3] = (P_G + 3 * P_H + 2) >> 2
    icp = blk - pred
    sae = compute_sae(icp)
    return icp, pred, sae


def pred_ddr_4_blk(blk, predpel):
    P_A, P_B, P_C, P_D = predpel[1:5]
    P_I, P_J, P_K, P_L = predpel[9:]
    P_X = predpel[0]
    pred = np.zeros((4, 4), dtype=int)
    pred[3][0] = (P_L + 2 * P_K + P_J + 2) >> 2
    pred[2][0] = pred[3][1] = (P_K + 2 * P_J + P_I + 2) >> 2
    pred[1][0] = pred[2][1] = pred[3][2] = (P_J + 2 * P_I + P_X + 2) >> 2
    pred[0][0] = pred[1][1] = pred[2][2] = pred[3][3] = (P_I + 2 * P_X + P_A + 2) >> 2
    pred[0][1] = pred[1][2] = pred[2][3] = (P_X + 2 * P_A + P_B + 2) >> 2
    pred[0][2] = pred[1][3] = (P_A + 2 * P_B + P_C + 2) >> 2
    pred[0][3] = (P_B + 2 * P_C + P_D + 2) >> 2
    icp = blk - pred
    sae = compute_sae(icp)
    return icp, pred, sae


def pred_vr_4_blk(blk, predpel):
    P_A, P_B, P_C, P_D = predpel[1:5]
    P_I, P_J, P_K, P_L = predpel[9:]
    P_X = predpel[0]
    pred = np.zeros((4, 4), dtype=int)
    pred[0][0] = pred[2][1] = (P_X + P_A + 1) >> 1
    pred[0][1] = pred[2][2] = (P_A + P_B + 1) >> 1
    pred[0][2] = pred[2][3] = (P_B + P_C + 1) >> 1
    pred[0][3] = (P_C + P_D + 1) >> 1
    pred[1][0] = pred[3][1] = (P_I + 2 * P_X + P_A + 2) >> 2
    pred[1][1] = pred[3][2] = (P_X + 2 * P_A + P_B + 2) >> 2
    pred[1][2] = pred[3][3] = (P_A + 2 * P_B + P_C + 2) >> 2
    pred[1][3] = (P_B + 2 * P_C + P_D + 2) >> 2
    pred[2][0] = (P_X + 2 * P_I + P_J + 2) >> 2
    pred[3][0] = (P_I + 2 * P_J + P_K + 2) >> 2
    icp = blk - pred
    sae = compute_sae(icp)
    return icp, pred, sae


def pred_hd_4_blk(blk, predpel):
    P_A, P_B, P_C, P_D = predpel[1:5]
    P_I, P_J, P_K, P_L = predpel[9:]
    P_X = predpel[0]
    pred = np.zeros((4, 4), dtype=int)
    pred[0][0] = pred[1][2] = (P_X + P_I + 1) >> 1
    pred[0][1] = pred[1][3] = (P_I + 2 * P_X + P_A + 2) >> 2
    pred[0][2] = (P_X + 2 * P_A + P_B + 2) >> 2
    pred[0][3] = (P_A + 2 * P_B + P_C + 2) >> 2
    pred[1][0] = pred[2][2] = (P_I + P_J + 1) >> 1
    pred[1][1] = pred[2][3] = (P_X + 2 * P_I + P_J + 2) >> 2
    pred[2][0] = pred[3][2] = (P_J + P_K + 1) >> 1
    pred[2][1] = pred[3][3] = (P_I + 2 * P_J + P_K + 2) >> 2
    pred[3][0] = (P_K + P_L + 1) >> 1
    pred[3][1] = (P_J + 2 * P_K + P_L + 2) >> 2
    icp = blk - pred
    sae = compute_sae(icp)
    return icp, pred, sae


def pred_vl_4_blk(blk, predpel):
    P_A, P_B, P_C, P_D = predpel[1:5]
    P_E, P_F, P_G, P_H = predpel[5:9]
    pred = np.zeros((4, 4), dtype=int)
    pred[0][0] = (P_A + P_B + 1) >> 1
    pred[0][1] = pred[2][0] = (P_B + P_C + 1) >> 1
    pred[0][2] = pred[2][1] = (P_C + P_D + 1) >> 1
    pred[0][3] = pred[2][2] = (P_D + P_E + 1) >> 1
    pred[2][3] = (P_E + P_F + 1) >> 1
    pred[1][0] = (P_A + 2 * P_B + P_C + 2) >> 2
    pred[1][1] = pred[3][0] = (P_B + 2 * P_C + P_D + 2) >> 2
    pred[1][2] = pred[3][1] = (P_C + 2 * P_D + P_E + 2) >> 2
    pred[1][3] = pred[3][2] = (P_D + 2 * P_E + P_F + 2) >> 2
    pred[3][3] = (P_E + 2 * P_F + P_G + 2) >> 2
    icp = blk - pred
    sae = compute_sae(icp)
    return icp, pred, sae


def pred_hu_4_blk(blk, predpel):
    P_I, P_J, P_K, P_L = predpel[9:]
    pred = np.zeros((4, 4), dtype=int)
    pred[0][0] = (P_I + P_J + 1) >> 1
    pred[0][1] = (P_I + 2 * P_J + P_K + 2) >> 2
    pred[0][2] = pred[1][0] = (P_J + P_K + 1) >> 1
    pred[0][3] = pred[1][1] = (P_J + 2 * P_K + P_L + 2) >> 2
    pred[1][2] = pred[2][0] = (P_K + P_L + 1) >> 1
    pred[1][3] = pred[2][1] = (P_K + 2 * P_L + P_L + 2) >> 2
    pred[3][0] = pred[2][2] = pred[2][3] = pred[3][1] = pred[3][2] = pred[3][3] = P_L
    icp = blk - pred
    sae = compute_sae(icp)
    return icp, pred, sae


def no_pred_4(Seq, i, j):
    icp = Seq[i:i + 4, j:j + 4]
    pred = np.zeros((4, 4), dtype=int)
    sae = compute_sae(icp)
    return icp, pred, sae


def mode_select_4_blk(blk, predpel, weights=1):
    icp1, pred1, sae1 = pred_vert_4_blk(blk, predpel)
    icp2, pred2, sae2 = pred_horz_4_blk(blk, predpel)
    icp3, pred3, sae3 = pred_dc_4_blk(blk, predpel)
    icp4, pred4, sae4 = pred_ddl_4_blk(blk, predpel)
    icp5, pred5, sae5 = pred_ddr_4_blk(blk, predpel)
    icp6, pred6, sae6 = pred_vr_4_blk(blk, predpel)
    icp7, pred7, sae7 = pred_hd_4_blk(blk, predpel)
    icp8, pred8, sae8 = pred_vl_4_blk(blk, predpel)
    icp9, pred9, sae9 = pred_hu_4_blk(blk, predpel)

    sae_values = [sae1, sae2, sae3, sae4, sae5, sae6, sae7, sae8, sae9]
    min_sae_index = np.argmin(sae_values)
    sae = sae_values[min_sae_index]
    modes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    mode = modes[min_sae_index]

    if mode == 0:
        icp = icp1
        pred = pred1
    elif mode == 1:
        icp = icp2
        pred = pred2
    elif mode == 2:
        icp = icp3
        pred = pred3
    elif mode == 3:
        icp = icp4
        pred = pred4
    elif mode == 4:
        icp = icp5
        pred = pred5
    elif mode == 5:
        icp = icp6
        pred = pred6
    elif mode == 6:
        icp = icp7
        pred = pred7
    elif mode == 7:
        icp = icp8
        pred = pred8
    elif mode == 8:
        icp = icp9
        pred = pred9
    return icp, pred, sae, mode


def no_pred_blk(blk):
    icp = blk
    pred = np.zeros_like(blk, dtype=int)
    sae = compute_sae(icp)
    return icp, pred, sae


def no_pred_16(Seq, i, j):
    icp = Seq[i:i + 16, j:j + 16]
    pred = np.zeros((16, 16), dtype=int)
    sae = compute_sae(icp)
    return icp, pred, sae


def predict_16_blk(blk, Seq_r, i, j, weights):
    if i == 0 and j == 0:  # No prediction
        mode = 4  # Special mode to describe no prediction
        icp, pred, sae = no_pred_blk(blk)
    elif i == 0:  # Horizontal prediction
        mode = 1
        icp_1, pred_1, sae_1 = pred_horz_16_blk(blk, Seq_r, i, j) 
        icp_2, pred_2, sae_2 = pred_horz_16_dc(blk, Seq_r, i, j)
        if sae_1 < sae_2:
            icp = icp_1
            pred = pred_1
            sae = sae_1
        else:
            icp = icp_2
            pred = pred_2
            sae = sae_2
            mode = 2
    elif j == 0:  # Vertical prediction
        mode = 0
        icp_1, pred_1, sae_1 = pred_vert_16_blk(blk, Seq_r, i, j)
        icp_2, pred_2, sae_2 = pred_vert_16_dc(blk, Seq_r, i, j)
        if sae_1 < sae_2:
            icp = icp_1
            pred = pred_1
            sae = sae_1
        else:
            icp = icp_2
            pred = pred_2
            sae = sae_2
            mode = 2
    else:  # Try all different prediction
        icp, pred, sae, mode = mode_select_16_blk(blk, Seq_r, i, j, weights)  
    return icp, pred, sae, mode


def pred_horz_16_dc(blk, Seq_r, i, j):
    pred = np.mean(Seq_r[i:i+16, j-1])
    icp = blk - pred
    sae = compute_sae(icp)
    return icp, pred, sae

def pred_vert_16_dc(blk, Seq_r, i, j):
    pred = np.mean(Seq_r[i-1, j:j+16])
    icp = blk - pred
    sae = compute_sae(icp)
    return icp, pred, sae

# 16x16 Horizontal prediction
def pred_horz_16_blk(blk, Seq_r, i, j):
    pred = Seq_r[i:i+16, j-1].reshape(16, 1).dot(np.ones((1, 16)))
    icp = blk - pred
    sae = compute_sae(icp)
    return icp, pred, sae

# 16x16 Vertical prediction
def pred_vert_16_blk(blk, Seq_r, i, j):
    pred = np.ones((16, 1)).dot(Seq_r[i-1, j:j+16].reshape(1, 16))
    icp = blk - pred
    sae = compute_sae(icp)
    return icp, pred, sae

# 16x16 DC prediction
def pred_dc_16_blk(blk, Seq_r, i, j):
    pred = (np.sum(Seq_r[i-1, j:j+16]) + np.sum(Seq_r[i:i+16, j-1]) + 16).astype(int) >> 5
    icp = blk - pred
    sae = compute_sae(icp)
    return icp, pred, sae


def pred_plane_16_blk(blk, Seq_r, i, j):
    H = np.sum((np.arange(8) + 1) * (Seq_r[i + np.arange(8) + 8, j - 1] - Seq_r[i + 6 - np.arange(8), j - 1]))
    V = np.sum((np.arange(8) + 1) * (Seq_r[i - 1, j + np.arange(8) + 8] - Seq_r[i - 1, j + 6 - np.arange(8)]))

    H = H.astype(int)
    V = V.astype(int)

    a = 16 * (Seq_r[i - 1, j + 15] + Seq_r[i + 15, j - 1])
    b = (5*H + 32) >> 6
    c = (5*V + 32) >> 6

    pred = np.empty((16, 16), dtype=int)

    for m in range(16):
        for n in range(16):
            d = (a + b * (m - 7) + c * (n - 7) + 16).astype(int) >> 5
            pred[m, n] = max(0, min(255, d))
    icp = blk - pred
    sae = compute_sae(icp)
    return icp, pred, sae


# Mode selection for 16x16 prediction
def mode_select_16_blk(blk, Seq_r, i, j, weights):
    icp1, pred1, sae1 = pred_vert_16_blk(blk, Seq_r, i, j)
    icp2, pred2, sae2 = pred_horz_16_blk(blk, Seq_r, i, j)
    icp3, pred3, sae3 = pred_dc_16_blk(blk, Seq_r, i, j)
    icp4, pred4, sae4 = pred_plane_16_blk(blk, Seq_r, i, j)

    sae_values = [sae1, sae2, sae3, sae4]
    min_sae_idx = sae_values.index(min(sae_values))

    modes = {
        0: (sae1, icp1, pred1),
        1: (sae2, icp2, pred2),
        2: (sae3, icp3, pred3),
        3: (sae4, icp4, pred4)
    }

    sae, icp, pred = modes[min_sae_idx]
    mode = min_sae_idx
    return icp, pred, sae, mode
