import numpy as np
import os
import nibabel as nib
from scipy.ndimage import (
    gaussian_filter, label, binary_fill_holes, binary_closing, generate_binary_structure,binary_opening, find_objects
)
from skimage.filters import threshold_otsu
from flyseg import file_read
# from Image_view_v3 import Image_viewer_simple
from sklearn.mixture import GaussianMixture

def fill_holes_by_slice(mask):
    """沿 Z 轴方向逐 slice 填洞并保留最大连通区域，同时记录面积突变"""
    # 自动将最长轴作为 Z 轴（最后一维）
    if mask.shape[0] >= mask.shape[1] and mask.shape[0] >= mask.shape[2]:
        mask = np.transpose(mask, (1, 2, 0))
    elif mask.shape[1] >= mask.shape[0] and mask.shape[1] >= mask.shape[2]:
        mask = np.transpose(mask, (0, 2, 1))
    # 否则保持原样

    filled_mask = np.zeros_like(mask, dtype=np.uint8)
    previous_area = None
    needs_check = False

    z_sums = [np.sum(mask[:, :, z]) for z in range(mask.shape[2])]
    non_empty_slices = [z for z, s in enumerate(z_sums) if s > 0]
    process_range = non_empty_slices[5:-5] if len(non_empty_slices) >= 21 else []

    structure = generate_binary_structure(2, 2)
    for z in range(mask.shape[2]):
        slice_2d = mask[:, :, z]
        processed = binary_closing(slice_2d, structure=structure)
        processed = binary_opening(processed, structure=structure)
        filled_2d = binary_fill_holes(processed).astype(np.uint8)

        labeled, num_features = label(filled_2d)
        if num_features == 0:
            continue
        elif num_features == 1 or z not in process_range:
            largest_region = filled_2d
        else:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            largest_label = np.argmax(sizes)
            largest_region = (labeled == largest_label).astype(np.uint8)

            area = int(np.sum(largest_region))
            if previous_area and previous_area > 0:
                if abs(area - previous_area) / previous_area > 2:
                    needs_check = True
            previous_area = area

        filled_mask[:, :, z] = largest_region

    return filled_mask, needs_check

def apply_otsu(image, low, high):
    """
    使用 Otsu 方法阈值化图像
    """
    data = image[(image > low) & (image < high)]
    thresh = threshold_otsu(data) if data.size > 0 else low
    image[image < thresh] = 0
    return image

# def fill_mask_holes(mask):
#     """
#     使用 3D 结构元素进行两次闭运算 + 填洞
#     """
#     struct_elem = generate_binary_structure(3, 2)  # 6-连通结构
#     closed_mask = binary_closing(mask, footprint=struct_elem)
#     closed_mask = binary_closing(closed_mask, footprint=struct_elem)
#     filled_mask = binary_fill_holes(closed_mask).astype(np.uint8)
#     return filled_mask

def apply_otsu_and_fill(img, intensity_min, intensity_max):
    img_array = img.astype(np.uint16).copy()
    img_process = apply_otsu(img_array, intensity_min, intensity_max)
    binary = (img_process > 0).astype(np.uint8)

    labeled, _ = label(binary)
    sizes = np.bincount(labeled.ravel())
    if len(sizes) <= 1:
        return binary, False

    label_max = sizes[1:].argmax() + 1
    region = (labeled == label_max).astype(np.uint8)
    objects = find_objects(region)

    if objects:
        # 选择最大的区域
        max_region = max(objects, key=lambda bbox: np.prod([s.stop - s.start for s in bbox]))
        z_min, z_max = max_region[0].start, max_region[0].stop
        y_min, y_max = max_region[1].start, max_region[1].stop
        x_min, x_max = max_region[2].start, max_region[2].stop

        print(f"Bounding box: z:[{z_min}, {z_max}], y:[{y_min}, {y_max}], x:[{x_min}, {x_max}]")

        # 为防止索引越界，设置切片边界
        z_start = max(z_min - 10, 0)
        z_end = min(z_max + 10, img_array.shape[0])
        y_start = max(y_min - 10, 0)
        y_end = min(y_max + 10, img_array.shape[1])
        x_start = max(x_min - 10, 0)
        x_end = min(x_max + 10, img_array.shape[2])

        # 使用调整后的索引进行切割
        processed_image = img_array[z_start:z_end, y_start:y_end, x_start:x_end]
        bbox = (z_start, z_end, y_start, y_end, x_start, x_end)
        # processed_image = image[:,:, x_start:x_end]
    else:
        # 如果没有找到区域，则直接返回原始图像
        processed_image = img_array
        bbox = None

    return processed_image, bbox

def restore_to_original_size(img, cropped_mask, bbox):
    """
    将 GMM 生成的裁剪区域 mask 重新放回原始 img 坐标系中。

    参数:
        img (numpy.ndarray): 原始 3D 图像
        cropped_mask (numpy.ndarray): GMM 生成的裁剪区域 mask
        bbox (tuple): (z_start, z_end, y_start, y_end, x_start, x_end) - 裁剪区域

    返回:
        new_mask (numpy.ndarray): 具有原始图像大小的 mask
    """
    new_mask = np.zeros_like(img, dtype=np.uint8)  # 创建与原始 img 大小相同的空 mask
    z_start, z_end, y_start, y_end, x_start, x_end = bbox  # 解析 bounding box
    new_mask[z_start:z_end, y_start:y_end, x_start:x_end] = cropped_mask  # 放回原坐标
    return new_mask

def fill_holes_by_slice_x(mask):
    """沿 X 轴方向逐 slice 填洞并保留最大连通区域"""
    filled_mask = np.zeros_like(mask, dtype=np.uint8)
    # structure = generate_binary_structure(2, 2)

    for z in range(mask.shape[0]):
        slice_2d = mask[z, :, :]
        # processed = binary_closing(slice_2d, structure=structure)
        # processed = binary_opening(processed, structure=structure)
        filled_2d = binary_fill_holes(slice_2d).astype(np.uint8)

        # labeled, num_features = label(filled_2d)
        # if num_features == 0:
        #     continue
        # sizes = np.bincount(labeled.ravel())
        # sizes[0] = 0
        # largest_label = np.argmax(sizes)
        # largest_region = (labeled == largest_label).astype(np.uint8)

        filled_mask[z, :, :] = filled_2d

    return filled_mask


def fill_holes_zx(mask):
    """
    综合处理：先沿 Z 轴 slice-wise 处理，再沿 X 轴 slice-wise 处理。
    返回：最终填充后的 mask 及 Z 轴是否存在跳变。
    """
    #
    mask_zx = fill_holes_by_slice_x(mask)
    mask_z, needs_check = fill_holes_by_slice(mask_zx)
    return mask_z, needs_check

def clean_filename(filename):
    """
    清除尾缀 _bodymask.nii.gz 和 .dcimg
    """
    name = filename.replace("_bodymask", "")
    name = name.replace(".dcimg", "")
    return name

def data_save_body(mask, save_dir, filename, dtype=np.uint8):
    """
    保存 3D mask 为 NIfTI 文件 (.nii.gz)
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{filename}.nii.gz")
    # print(f"✅ Saving file to: {save_path}")  # <—— 必须添加这一行！
    nii_img = nib.Nifti1Image(mask.astype(dtype), affine=np.eye(4))
    nib.save(nii_img, save_path)
    # print(f"✅ Saved: {save_path}")

def apply_GMM(image, components = 3, body_idx =-2):
    """
    使用 Otsu 方法阈值化图像
    """
    cropped, bbox = apply_otsu_and_fill(image, 50, 1000)
    gmm = GaussianMixture(n_components=components, covariance_type="full", random_state=42, max_iter=100)
    img_flat = cropped.flatten().reshape(-1, 1)
    gmm.fit(img_flat)
    labels = gmm.predict(img_flat).reshape(cropped.shape)
    means = gmm.means_.flatten()
    sorted_indices = np.argsort(means)  # 返回排序后的索引
    component = sorted_indices[body_idx]
    mask1 = (labels == component).astype(np.uint8)
    mask = restore_to_original_size(image, mask1, bbox)
    return mask

def post_mask(mask):
    """
    进一步平滑并保留最大连通区域
    """
    mask = (mask > 0).astype(np.uint8)
    mask = binary_fill_holes(mask).astype(np.uint8)
    smoothed = gaussian_filter(mask.astype(np.float32), sigma=2)
    smoothed_binary = (smoothed > 0.5).astype(np.uint8)
    # filled_mask = binary_fill_holes(smoothed_binary).astype(np.uint8)

    def remove_largest(mask, sigma):
        labeled_mask, num_features = label(mask)
        if num_features == 0:
            return mask

        sizes = np.bincount(labeled_mask.ravel())
        sizes[0] = 0  # 忽略背景
        largest = np.argmax(sizes)
        largest_mask = (labeled_mask == largest).astype(np.uint8)
        smoothed = gaussian_filter(largest_mask.astype(np.float32), sigma=sigma)
        return (smoothed > 0.5).astype(np.uint8)

    mask = remove_largest(smoothed_binary, 2)
    mask = remove_largest(mask, 2)
    return mask

def process_file(file_path, output_folder, components = 3, body_idx = -2):
    try:
        img = file_read(file_path)
        mask = apply_GMM(img, components, body_idx)
        mask1 = post_mask(mask)
        mask2, needs_check = fill_holes_zx(mask1)
        mask3 = post_mask(mask2)
        voxel_count = int(np.count_nonzero(mask3))
        # print(f"mask2 dtype: {mask2.dtype}, shape: {mask2.shape}, unique values: {np.unique(mask2)}")
        filename = os.path.splitext(os.path.basename(file_path))[0] + "_bodymask"
        data_save_body(mask3, output_folder, filename)
        print(f"save {filename} into {output_folder}")
        cleaned_name = clean_filename(filename)
        return {
            "filename": cleaned_name,
            "voxel_count": voxel_count,
            "needs_check": needs_check,
            "status": "success"
        }
    except Exception as e:
        return {
            "filename": os.path.basename(file_path),
            "status": "error",
            "error": str(e)
        }