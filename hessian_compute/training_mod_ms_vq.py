import numpy as np
from STAC_AVC.utils_avc import read_image_resize_rect
import os
import matplotlib.pyplot as plt
from STAC_AVC.hessian_compute.compute_Q_jax import compute_Q_ssim as compute_Q
from STAC_AVC.hessian_compute.q_ops_gain_shape import q_ops_ssim as q_ops
from sklearn.cluster import KMeans
import random

N = 8
n_cwd = 8
true_N = (512, 512)
nqs = 6

compute_Q_obj = compute_Q(true_N=true_N, sampling_depth=16)
q_ops_obj = q_ops(true_N=true_N, N=N, nqs=nqs)

path = 'Images/CLIC/Training/'
dirs = os.listdir(path)
num_images = 100
random.seed(0)
random.shuffle(dirs)
dirs = dirs[:num_images]

q_vecs_8 = np.zeros((len(dirs), true_N[0]//N, true_N[1]//N, N*N))

ind_image = -1
for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()
    Q = compute_Q_obj.sample_q(img)
    Q = q_ops_obj.normalize_q(Q)

    if np.isnan(Q).any():
        print('Nan in Q')
        continue

    for i in range(0, true_N[0]//N):
        for j in range(0, true_N[1]//N):
            ravel_q = Q[i:i+N, j:j+N].ravel('F')
            q_vecs_8[ind_image, i, j, :] = np.array(ravel_q)

q_vecs_8_tmp = q_vecs_8
target = q_ops_obj.name_target()

q_vecs_8 = q_vecs_8_tmp.transpose(3, 0, 1, 2)
q_vecs_8 = q_vecs_8[~np.isnan(q_vecs_8).any(axis=(1, 2, 3))]
q_batch = q_vecs_8.reshape(N**2, -1)

mean_vals = np.mean(q_batch, axis=0)
n_cwds_mean = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
mean_list = []

mean_vals = mean_vals.reshape(1, -1)
for n_cwd in n_cwds_mean:
    # kmeans clustering
    kmeans = KMeans(n_clusters=n_cwd, random_state=0).fit(mean_vals.T)
    # compute the centroids
    centroids = kmeans.cluster_centers_
    mean_list.append(centroids)


n_cwds_shape = [1, 2, 3]
shape_list = []

q_vecs_8 = q_vecs_8_tmp.transpose(3, 0, 1, 2)
q_vecs_8 = q_vecs_8[~np.isnan(q_vecs_8).any(axis=(1, 2, 3))]
q_batch = q_vecs_8.reshape(N**2, -1)

for ind_shape in range(len(n_cwds_shape)):
    for ind_mean in range(len(n_cwds_mean)):
        # kmeans clustering

        mean_clusters = mean_list[ind_mean]
        mean_clusters = mean_clusters.reshape(1, -1)
            
        # use mean_vals to find the closest values to the mean clusters
        q_vecs = np.zeros_like(q_batch)
        for i in range(q_batch.shape[1]): 
            pos_min = np.argmin(np.abs(mean_clusters[0] - mean_vals[0][i]))
            q_vecs[:, i] = q_batch[:, i] - mean_clusters[0][pos_min]

        kmeans = KMeans(n_clusters=n_cwds_shape[ind_shape], random_state=0).fit(q_vecs.T)
        # compute the centroids
        centroids = kmeans.cluster_centers_
        centroids = centroids.reshape(n_cwds_shape[ind_shape], N, N, order='F')
        shape_list.append(centroids)

        final_centroids = np.zeros((N, N, n_cwds_mean[ind_mean]*n_cwds_shape[ind_shape]))
        for j in range(n_cwds_shape[ind_shape]):
            for i in range(n_cwds_mean[ind_mean]):
                final_centroids[:, :, j*n_cwds_mean[ind_mean]+i] = centroids[j, :, :] + mean_clusters[0][i]

        n_cwd_mean = n_cwds_mean[ind_mean]
        n_cwd_shape = n_cwds_shape[ind_shape]

        str_save = 'week_5/data/centroids/centroids_' + target + '_' + str(n_cwd_mean) + '_' + str(n_cwd_shape) + '_' + str(true_N) + '_' + str(N) + '.npy'
        centroids = final_centroids.transpose(2, 0, 1)

        np.save(str_save, centroids)
        plt.figure()
        for i in range(centroids.shape[0]):
            plt.subplot(1, centroids.shape[0], i+1)
            plt.imshow(centroids[i, :, :], cmap='gray')
            plt.title('Centroid '+str(i), fontsize=16)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
plt.show()