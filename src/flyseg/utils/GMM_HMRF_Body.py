import numpy as np
from scipy.ndimage import (gaussian_filter, label, binary_fill_holes)
from sklearn.mixture import GaussianMixture
from numba import njit, prange
import os
import nibabel as nib

def mask_save(mask, save_dir, filename, dtype=np.uint8):
    """
    将 3D mask 保存为 NIfTI 文件 (.nii.gz)

    参数：
    - mask: numpy ndarray，3D 二值或多类分割 mask
    - save_dir: 输出文件夹路径
    - filename: 不带扩展名的保存名（如 'sample1_mask'）
    - dtype: 可选保存类型，默认 uint8
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{filename}")

    # 创建 NIfTI 图像对象，affine 默认为单位矩阵
    nii_img = nib.Nifti1Image(mask.astype(dtype), affine=np.eye(4))
    nib.save(nii_img, save_path)
    # print(f"✅ Saved NIfTI mask: {save_path}")

class Body_processor:
    @staticmethod
    def restore_mask(shape, mask, bbox):
        restored = np.zeros(shape, dtype=np.uint8)
        z1, z2, y1, y2, x1, x2 = bbox
        restored[z1:z2, y1:y2, x1:x2] = mask
        return restored

    @staticmethod
    def post_mask(mask):
        mask = (mask > 0).astype(np.uint8)
        mask = binary_fill_holes(mask)
        smoothed = gaussian_filter(mask.astype(np.float32), sigma=2)
        smoothed_binary = (smoothed > 0.5).astype(np.uint8)
        filled_mask = binary_fill_holes(smoothed_binary).astype(np.uint8)
        def remove_objects(mask, sigma):
            labeled_mask, num_features = label(mask)
            # 如果没有任何组件，则直接返回原始 mask
            if num_features == 0:
                print("error")

            # 统计各个连通组件的像素数量
            component_sizes = np.bincount(labeled_mask.ravel())
            # 将背景（label为0）的数量设为0，防止选到背景
            component_sizes[0] = 0

            # 找到最大组件的标签（索引）
            largest_component = np.argmax(component_sizes)

            # 构造新的 mask：仅保留最大组件，将其余部分设为0
            img1 = (labeled_mask == largest_component).astype(np.uint8)
            img2 = gaussian_filter(img1.astype(np.float32), sigma=sigma)
            img3 = (img2 > 0.5).astype(np.uint8)
            return img3
        mask = remove_objects(filled_mask, 1)
        return mask

class GMM_HMRF_body:
    def __init__(self, n_components =3, alpha =0.5, eps = 1e-3, max_iter = 10, neighborhood = 26):
        self.n_components = n_components
        self.alpha = alpha
        self.eps = eps
        self.max_iter = max_iter
        self.neighborhood = neighborhood

    def get_offsets(self):
        if self.neighborhood == 6:
            offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        elif self.neighborhood == 18:
            offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1),
                       (-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0),
                       (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1),
                       (-1, 0, -1), (-1, 0, 1), (1, 0, -1), (1, 0, 1)]
        elif self.neighborhood == 26:
            offsets = [(dz, dy, dx) for dz in [-1, 0, 1] for dy in [-1, 0, 1] for dx in [-1, 0, 1]
                       if not (dz == dy == dx == 0)]
        else:
            raise ValueError("neighborhood must be 6, 18 or 26")
        return np.array(offsets, dtype=np.int8)  # shape (N, 3)

    @staticmethod
    @njit(parallel=True)
    def _icm(volume, labels, means, vars, offsets, alpha, target):
        Z, Y, X = volume.shape
        new_labels = labels.copy()
        for z in prange(Z):
            for y in range(Y):
                for x in range(X):
                    voxel = volume[z, y, x]
                    best_energy = 1e12
                    best_label = 0
                    for k in range(means.shape[0]):
                        nll = 0.5 * np.log(vars[k]) + ((voxel - means[k]) ** 2) / (2 * vars[k])
                        penalty = 0
                        for i in range(offsets.shape[0]):
                            dz, dy, dx = offsets[i]
                            zz, yy, xx = z + dz, y + dy, x + dx
                            if 0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X:
                                if labels[zz, yy, xx] != k:
                                    penalty += 1
                            if isinstance(target, (list, tuple, set)):
                                if k in target:
                                    penalty *= 0.75
                            else:
                                if k == target:
                                    penalty *= 0.75
                        total_energy = nll + alpha * penalty
                        if total_energy < best_energy:
                            best_energy = total_energy
                            best_label = k
                    new_labels[z, y, x] = best_label
        return new_labels

    def _update_gmm(self, volume, labels, gmm):
        means, covs, weights = [], [], []
        N = volume.size
        for k in range(self.n_components):
            voxels = volume[labels == k]
            if voxels.size > 0:
                means.append(np.mean(voxels))
                covs.append(max(np.var(voxels), 1e-8))
                weights.append(voxels.size / N)
            else:
                means.append(gmm.means_[k][0])
                covs.append(max(gmm.covariances_[k][0, 0], 1e-8))
                weights.append(1.0 / self.n_components)
        gmm.means_ = np.array(means).reshape(-1, 1)
        gmm.covariances_ = np.array(covs).reshape(-1, 1, 1)
        gmm.weights_ = np.array(weights)
        return gmm

    def gmm_fit(self, volume):
        data = volume.reshape(-1, 1)
        gmm = GaussianMixture(n_components=self.n_components, covariance_type='full', random_state=42)
        gmm.fit(data)
        labels = gmm.predict(data).reshape(volume.shape)
        prev = labels.copy()
        for _ in range(self.max_iter):
            target = np.argsort(gmm.means_.flatten())[-2]
            labels = self._icm(volume, prev, gmm.means_.flatten(), gmm.covariances_.flatten(), self.get_offsets(),
                               self.alpha, target)
            if np.mean(labels != prev) < self.eps:
                break
            prev = labels.copy()
            gmm = self._update_gmm(volume, labels, gmm)
        self.gmm = gmm
        # 选取 GMM 均值最大的两个 label 索引
        sorted_indices = np.argsort(gmm.means_.flatten())
        target1 = sorted_indices[-1]
        target2 = sorted_indices[-2]
        # 合并这两个标签作为前景
        merged_mask = np.zeros_like(labels, dtype=np.uint8)
        merged_mask[labels == target1] = 2
        merged_mask[labels == target2] = 1
        # tissue_mask = ((labels == target1) | (labels == target2)).astyp
        # e(np.uint8)
        return merged_mask, gmm.means_, gmm.covariances_, gmm.weights_
