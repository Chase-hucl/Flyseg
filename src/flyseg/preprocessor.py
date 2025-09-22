import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import (
    gaussian_filter,
    label,
    binary_fill_holes,
    find_objects
)
from sklearn.mixture import GaussianMixture
from skimage.filters import threshold_otsu
from flyseg.utils.flySeg_reader import file_read, data_save
from flyseg.utils.GMM_Body import restore_to_original_size, post_mask, fill_holes_zx, clean_filename, data_save_body
# from flyseg.utils.GMM_HMRF_Body import mask_save, Body_processor,GMM_HMRF_body
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Tuple, List, Any, Dict
import csv
# from flyseg.utils.Otsu_seg_body import fill_holes_by_slice, post_mask

def apply_otsu(image: np.ndarray, lower_bound: float, upper_bound: float) -> Tuple[np.ndarray, float]:
    """
    Applies Otsu's thresholding within a specified intensity range.

    Args:
        image: Input 3D image.
        lower_bound: Lower intensity bound.
        upper_bound: Upper intensity bound.

    Returns:
        thresholded image and Otsu threshold value.
    """
    image = image.copy()
    data = image[(image > lower_bound) & (image < upper_bound)]
    thresh = threshold_otsu(data) if data.size > 0 else lower_bound
    image[image < thresh] = 0
    return image, thresh

def process_image(image_path: str, sigma: float = 2.0) -> Tuple[np.ndarray, float, np.ndarray, Tuple[int, int, int, int, int, int]]:
    """
    Processes a 3D image: thresholding, smoothing, and extracting largest component.

    Args:
        image_path: Path to input image.
        sigma: Standard deviation for Gaussian filter.

    Returns:
        Tuple of processed image, threshold, and binary mask.
    """
    image = file_read(image_path).astype(np.uint16)
    # image = image.copy()
    image_1, threshold = apply_otsu(image, 20, 1000)
    binary_mask = (image_1 > 0).astype(np.uint8)
    labeled_image, _ = label(binary_mask)
    component_sizes = np.bincount(labeled_image.ravel())
    largest_component_label = component_sizes[1:].argmax() + 1
    largest_component_mask = (labeled_image == largest_component_label)
    filled_mask = binary_fill_holes(largest_component_mask)
    bounding_boxes = find_objects(filled_mask)
    final_mask = np.zeros_like(image, dtype=np.uint8)

    if bounding_boxes:
        bbox = max(bounding_boxes, key=lambda box: np.prod([s.stop - s.start for s in box]))
        z_min, z_max = bbox[0].start, bbox[0].stop
        y_min, y_max = bbox[1].start, bbox[1].stop
        x_min, x_max = bbox[2].start, bbox[2].stop
        final_mask[z_min:z_max, y_min:y_max, x_min:x_max] = filled_mask[z_min:z_max, y_min:y_max, x_min:x_max]
        processed_image = image * final_mask
        z, y, x = bounding_boxes[0]
        boundingbox = (
            max(z.start - 10, 0), min(z.stop + 10, image.shape[0]),
            max(y.start - 10, 0), min(y.stop + 10, image.shape[1]),
            max(x.start - 10, 0), min(x.stop + 10, image.shape[2])
        )
        cropped = image[boundingbox[0]:boundingbox[1], boundingbox[2]:boundingbox[3], boundingbox[4]:boundingbox[5]]
    else:
        processed_image = image
        cropped = image.copy()
        boundingbox = (0, image.shape[0], 0, image.shape[1], 0, image.shape[2])
    return processed_image, threshold, cropped, boundingbox

def process_and_save_image(
    image_path: str,
    save_image_dir: str,
    body_dir: str,
) -> Dict[str, Any]:
    """
    处理单张图像并保存结果，同时返回处理信息。

    参数
    ----
    image_path : str
        输入图像路径
    save_image_dir : str
        处理后图像的保存目录
    body_dir : str
        body mask 保存目录
    alpha : float, optional
        GMM_HMRF 的正则化参数 (默认 2.0)

    返回
    ----
    Dict[str, Any]
        {
            "image_path": str,
            "image_save_path": str,
            "threshold": float,
            "image_shape": Tuple[int, int, int],
            "filename": str,
            "needs_check": bool,
            "status": str,
            "means": List[float],
            "covariances": List[float],
            "weights": List[float]
        }
    """
    # Step 1: 预处理图像
    processed_image, threshold, cropped, bbox = process_image(image_path)
    image_shape = tuple(processed_image.shape)

    # Step 2: 设置文件名和保存路径
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    image_filename = f"{base_filename}.nii.gz"
    image_save_path = os.path.join(save_image_dir, image_filename)

    # Step 3: GMM 聚类
    img_flat = cropped.flatten().reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=3,
        covariance_type="full",
        random_state=42,
        max_iter=100
    )
    gmm.fit(img_flat)

    labels = gmm.predict(img_flat).reshape(cropped.shape)
    means = gmm.means_.flatten().tolist()
    covariances = gmm.covariances_.flatten().tolist()
    weights = gmm.weights_.flatten().tolist()

    # Step 4: 选择感兴趣的成分（次大均值）
    sorted_indices = np.argsort(means)
    component = sorted_indices[-2]
    mask0 = (labels == component).astype(np.uint8)

    # Step 5: 恢复到原始大小
    mask = restore_to_original_size(processed_image, mask0, bbox)

    # Step 6: 保存处理后的图像
    data_save(processed_image, save_image_dir, image_filename, file_format='nii')

    # Step 7: mask 后处理
    mask1 = post_mask(mask)
    mask2, needs_check = fill_holes_zx(mask1)
    mask3 = post_mask(mask2)

    # Step 8: 保存 body mask
    filename = f"{base_filename}_bodymask"
    data_save_body(mask3, body_dir, filename)
    cleaned_name = clean_filename(filename)

    # Step 9: 返回结果
    return {
        "image_path": image_path,
        "image_save_path": image_save_path,
        "threshold": threshold,
        "image_shape": image_shape,
        "filename": cleaned_name,
        "needs_check": needs_check,
        "status": "success",
        "means": means,
        "covariances": covariances,
        "weights": weights
    }

def process_images_multithreaded(input_folder: str, save_image_dir: str, body_dir: str, max_workers: int = 3) -> List[Tuple]:
    file_info = []

    h5_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".h5") or file.endswith(".nii.gz"):
                h5_files.append(os.path.join(root, file))

    total_files = len(h5_files)
    print(f"🧾 Found {total_files} raw .h5 images in '{input_folder}'")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_and_save_image, image_path, save_image_dir, body_dir): image_path for image_path in h5_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            try:
                result = future.result()
                file_info.append(result)
            except Exception as e:
                print(f"❌ Failed to process image {futures[future]}: {e}")

    return file_info

def export_file_info_to_csv(
    file_info: List[Tuple[str, str, float, Tuple[int, int, int], str, List[float], List[float], List[float]]],
    csv_path: str,
    info_note: str = ""
) -> None:
    """
    Export processed image info to a CSV file.

    Args:
        file_info: List of tuples:
            (original_path, processed_path, threshold, shape, body_path, means, covariances, weights)
        csv_path: Output CSV file path.
        info_note: Optional extra info column.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Info",
            "Original Path",
            "Processed Path",
            "Threshold",
            "Shape (Z,Y,X)",
            "Body Path",
            "Means",
            "Covariances",
            "Weights"
        ])

        for entry in file_info:
            writer.writerow([
                info_note,
                entry["image_path"],
                entry["image_save_path"],
                entry["threshold"],
                str(entry["image_shape"]),
                entry["filename"],  # 或者 entry["image_save_path"] 看你想要什么
                ";".join(map(lambda x: f"{x:.6f}", entry["means"])),
                ";".join(map(lambda x: f"{x:.6f}", entry["covariances"])),
                ";".join(map(lambda x: f"{x:.6f}", entry["weights"]))
            ])




