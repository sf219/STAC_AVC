import numpy as np
import scipy 
from STAC_AVC.utils_lpit import rlgr, dpcm_smart
from STAC_AVC.hessian_compute.toolkit_iagft import get_transform_basis
import scipy.stats


def filter_q(Q, sigma=2):
    truncate = 3.5
    sigma = sigma
    r = int(truncate * sigma + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    
    pad = (win_size-1)//2
    #Q = Q.reshape(self.true_N[0], self.true_N[1], order='F')
    img1 = np.pad(Q, pad, mode='symmetric')

    x = np.linspace(-truncate, truncate, win_size)
    window = scipy.stats.norm.pdf(x, scale=sigma) * scipy.stats.norm.pdf(x[:, None], scale=sigma)
    window = window/np.sum(window)
    
    Q_filt = scipy.signal.convolve(img1, window, mode='valid')
    return Q_filt


class q_ops():

    def __init__(self, true_N, N=4, nqs=6):
        self.true_N = true_N
        self.N = N
        self.nqs = nqs
        self.n_cwds_or_means = [1, 2, 3, 4, 5, 6]
        self.n_cwds_or_shapes = [1, 2, 3]
        self.n_cwds = [(1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3), (4, 1), (4, 2), (5, 1), (6, 1)]
        #self.n_cwds = [(4, 1)]

    def quantize_q(self, new_Q, img):
        n_cwds_means = self.n_cwds_or_means.copy()
        n_cwds_shapes = self.n_cwds_or_shapes.copy()
        centroids_list = []
        ind_closest_list = []
        ind_closest_420_list = []
        Q_list = []
        rate_list = []
        SSE_list = []
        NNF_list = []
        true_N = self.true_N
        N = self.N

        self.set_mult(new_Q)

        indis = -1
        for n_cwd in self.n_cwds:
            indis += 1
            name_target = self.name_target()
            str_load = 'STAC_AVC/hessian_compute/centroids/centroids_' + name_target + '_' + str(n_cwd[0]) + '_' + str(n_cwd[1]) + '_' + str(true_N) + '_' + str(N) + '.npy'
            centroids = np.load(str_load)
            centroids_scale = centroids
            U, _, _ = get_transform_basis(centroids_scale)
            for i in range(2):
                ind_closest = self.quantize_q_cen(new_Q, centroids_scale, U)
                centroids_scale, Q_scale, U = self.scale_centroids(centroids_scale, new_Q, ind_closest, U)
            
            centroids_list.append(centroids_scale)
            ind_closest_list.append(ind_closest)
            ind_closest_420 = self.reduce_420(ind_closest)
            ind_closest_420_list.append(ind_closest_420)
            Q_list.append(Q_scale)

            NNF = 0
            for i in range(0, true_N[0], N):
                for j in range(0, true_N[1], N):
                    Q_blk = new_Q[i:i+N, j:j+N].ravel('F')
                    Qmt = np.diag(Q_blk) #/ np.sum(Q_blk) * np.sum(Q_quan_blk)
                    index = ind_closest[i//N, j//N].astype(int)
                    U_index = U[index]
                    #NNF += np.sum(np.square(img_blk - U_index @ U_index.T @ Qmt @ img_blk))
                    trans = U_index.T @ Qmt @ U_index
                    NNF += np.sum(np.square(trans - np.eye(N**2)))
            SSE = np.sum(np.square(new_Q - Q_scale))
            NNF_list.append(NNF)
            SSE_list.append(SSE)
            rate, _ = self.compress_q(ind_closest, centroids.shape[0])
            rate_list.append(rate)

        self.centroids_list = centroids_list
        self.ind_closest_list = ind_closest_list
        self.ind_closest_420_list = ind_closest_420_list
        self.Q_list = Q_list
        self.rate_list = rate_list
        self.SSE_list = SSE_list
        self.n_cwds_means = n_cwds_means
        self.n_cwds_shapes = n_cwds_shapes
        self.NNF_list = NNF_list


    def choose_ncwd(self, lbr_mode=False):
        rdos = []

        lam = np.mean(np.asarray(self.NNF_list))/np.mean(np.asarray(self.rate_list))*self.mult

        for i in range(len(self.rate_list)):
            rdos.append(self.SSE_list[i]*0 + lam*self.rate_list[i] + self.NNF_list[i])
            print('rate: ', self.rate_list[i], 'SSE: ', self.SSE_list[i], 'NFF: ', self.NNF_list[i], 'RDO: ', rdos[i])
        print('Max values: ', self.max_val, 'Mult: ', self.mult)
        ind = np.argmin(rdos)
        self.ind_closest = self.ind_closest_list[ind]
        self.ind_closest_420 = self.ind_closest_420_list[ind]
        self.Q = self.Q_list[ind]
        self.centroids = self.centroids_list[ind]
        self.overhead_bits = self.rate_list[ind]

    def reduce_420(self, ind_closest):
        ind_closest_420 = np.zeros((ind_closest.shape[0]//2, ind_closest.shape[1]//2))
        # iterate in blocks of size 2 by 2
        for i in range(ind_closest.shape[0]//2):
            for j in range(ind_closest.shape[1]//2):
                tmp = ind_closest[2*i:2*i+2, 2*j:2*j+2]
                tmp = tmp.ravel('F').astype(int)    
                output = scipy.stats.mode(tmp, axis=0).mode
                ind_closest_420[i, j] = int(output)
        return ind_closest_420

    def quantize_q_cen(self, new_Q, centroids, U):
        n_cwd = centroids.shape[0]
        N = self.N
        ind_closest = np.zeros((self.true_N[0]//N, self.true_N[1]//N))
       
        def test_block(new_Q_blk):
            dists = np.zeros(n_cwd)
            for k in range(n_cwd):
                U_basis = U[k]
                Qmt = np.diag(new_Q_blk) / np.sum(new_Q_blk) * np.sum(centroids[k, :])
                dists[k] = np.sum(np.square(U_basis.T @ Qmt @ U_basis - np.eye(N**2)))
            return np.argmin(dists)
        
        for i in range(0, self.true_N[0], N):
            for j in range(0, self.true_N[1], N):
                new_Q_blk = new_Q[i:i+N, j:j+N].ravel('F')
                ind_min = test_block(new_Q_blk)
                ind_closest[i//N, j//N] = ind_min
        return ind_closest


    def scale_centroids(self, centroids, new_Q, ind_closest, U):
        #iterate over all the blocks of new_Q
        n_cwd = centroids.shape[0]
        N = self.N
        true_N = self.true_N
        avg_weight = np.zeros((n_cwd))
        counts = np.zeros((n_cwd))
        for i in range(0, true_N[0], N):
            for j in range(0, true_N[1], N):
                new_Q_blk = new_Q[i:i+N, j:j+N].ravel('F')
                ind = ind_closest[i//N, j//N].astype(int)
                avg_weight[ind] += np.mean(new_Q_blk)
                counts[ind] += 1
        avg_weight = avg_weight/counts
        avg_weight[np.isnan(avg_weight)] = 1
        ratio = np.zeros_like(avg_weight)
        for i in range(n_cwd):
            ratio[i] = avg_weight[i]/np.mean(centroids[i, :])
        #print('ratio: ', ratio)
        new_centroids = np.zeros(centroids.shape)
        for i in range(n_cwd):
            new_centroids[i, :] = centroids[i, :]*ratio[i]
            U[i] = U[i]/np.sqrt(ratio[i])
        
        Q = np.zeros_like(new_Q)
        for i in range(0, true_N[0], N):
            for j in range(0, true_N[1], N):
                Q[i:i+N, j:j+N] = new_centroids[ind_closest[i//N, j//N].astype(int), :]

        normalizer = Q.size / np.sum(Q)

        Q = Q * normalizer
        
        for i in range(n_cwd):
            new_centroids[i, :] = new_centroids[i, :] * normalizer
            U[i] = U[i] / np.sqrt(normalizer)

        return new_centroids, Q, U

    def get_centroids(self, quant_level=None):
        output = (self.centroids)
        return output
    
    def compress_q(self, ind_closest, n_cwd=None):
        ind_closest = ind_closest.copy()
        
        unique = np.unique(ind_closest).astype(int)
        probs = np.zeros((unique.shape[0]))
        for i in range(unique.shape[0]):
            probs[i] = np.mean(ind_closest == unique[i])
        or_ind_closest = ind_closest.copy()
        arg_inc = np.argsort(probs)[::-1]
        for i in range(unique.shape[0]):
            ind_closest[or_ind_closest == unique[arg_inc[i]]] = i

        fdpcm = dpcm_smart(ind_closest)
        bits = np.zeros((8))
        for l in range(8):
            bits[l] = len(rlgr(fdpcm.ravel('F').astype(np.int32), L=l+1))*8
        index_min = np.argmin(bits)
        byte_seq = rlgr(fdpcm.astype(np.int32), L=index_min+1)
        bits = np.min(bits)+3
        return bits, byte_seq

    def uncompress_q(self, byte_seq):
        pass

    def name_target(self):
        return 'mse'

    def normalize_q(self, Q):
        Q = Q.squeeze()
        #Q = Q + 0.01*np.max(Q)
        Q = Q / np.sum(Q) * Q.size
        return Q
    
    def set_mult(self, new_Q):
        self.max_val = np.max(new_Q)
        self.mult = 1


class q_ops_ssim(q_ops):

    def __init__(self, true_N, N, nqs):
        super().__init__(true_N, N, nqs)

    def normalize_q(self, Q):
        Q = Q.squeeze()
        Q = Q / np.sum(np.abs(Q)) * Q.size
        return Q

    def name_target(self):
        return 'ssim'
    
    def set_mult(self, new_Q):
        min_max = 0.5
        max_max = 5.75
        self.max_val = np.clip(np.max(new_Q), min_max, max_max)
        a = -0.375  # Adjust the coefficient to control the curvature of the parabola
        b = 2.25
        c = -0.5
        self.mult = 1 #2*(3-(a * self.max_val ** 2 + b * self.max_val + c))

    

class q_ops_lester_ssim(q_ops):

    def __init__(self, true_N, N, nqs):
        super().__init__(true_N, N, nqs)

    def normalize_q(self, Q):
        Q = Q.squeeze()
        Q = Q / np.sum(np.abs(Q)) * Q.size
        return Q

    def name_target(self):
        return 'lester_ssim'
    

class q_ops_msssim(q_ops):
    
    def __init__(self, true_N,  N, nqs):
        super().__init__(true_N,  N, nqs)


    def normalize_q(self, Q):
        Q = Q.squeeze()
        Q[Q < 0] = 0
        Q = Q + 0.01*np.max(Q)
        Q = Q / np.sum(np.abs(Q)) * Q.size
        medianQ = np.median(Q)
        Q[Q > 5*medianQ] = 5*medianQ
        Q = Q / np.sum(np.abs(Q)) * Q.size
        return Q

    def name_target(self):
        return 'msssim'


class q_ops_brisque(q_ops):
    
    def __init__(self, true_N,  N, nqs):
        super().__init__(true_N,  N, nqs)

    def name_target(self):
        return 'brisque'
    
    def normalize_q(self, Q):
        #Q = self.filter_Q(Q)
        Q = Q.squeeze()
        Q[Q < 0] = 0
        #Q = filter_q(Q, 1)
        Q = Q + 0.01*np.max(Q)
        Q = Q / np.sum(np.abs(Q)) * Q.size
        medianQ = np.median(Q)
        Q[Q > 3*medianQ] = 3*medianQ
        Q = Q / np.sum(np.abs(Q)) * Q.size
        return Q
    

class q_ops_niqe(q_ops):
    
    def __init__(self, true_N,  N, nqs):
        super().__init__(true_N,  N, nqs)

    def name_target(self):
        return 'brisque'
    
    def normalize_q(self, Q, sigma=1):
        #Q = self.filter_Q(Q)
        Q[np.isnan(Q)] = np.median(Q)
        Q = Q.squeeze()
        Q[Q < 0] = 0
        #Q = filter_q(Q, sigma)
        Q = Q + 0.01*np.max(Q)
        Q = Q / np.sum(np.abs(Q)) * Q.size
        medianQ = np.median(Q)
        Q[Q > 3*medianQ] = 3*medianQ
        Q = Q / np.sum(np.abs(Q)) * Q.size
        return Q


class q_ops_lpips(q_ops):
    
    def __init__(self, true_N,  N, nqs):
        super().__init__(true_N,  N, nqs)

    def name_target(self):
        return 'lpips'
    
    def normalize_q(self, Q, sigma=0.1):
        Q = Q.squeeze()
        Q[Q < 0] = 0
        #Q = filter_q(Q, sigma)
        Q = Q + 0.01*np.max(Q)
        #Q = Q + 0.01*np.max(Q)
        Q = Q / np.sum(np.abs(Q)) * Q.size
        medianQ = np.median(Q)
        Q[Q > 5*medianQ] = 5*medianQ
        Q = Q / np.sum(np.abs(Q)) * Q.size
        return Q
    

class q_ops_resnet(q_ops):

    def __init__(self, true_N,  N, nqs):
        super().__init__(true_N,  N, nqs)

    def name_target(self):
        return 'resnet'
    
    def normalize_q(self, Q):
        Q = Q.squeeze()
        Q[Q < 0] = 0
        Q = Q + 0.01*np.max(Q)
        Q = Q / np.sum(np.abs(Q)) * Q.size
        medianQ = np.median(Q)
        Q[Q > 5*medianQ] = 5*medianQ
        Q = Q / np.sum(np.abs(Q)) * Q.size
        return Q