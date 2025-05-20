import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import (
    gaussian_filter,
    label,
    binary_fill_holes,
    find_objects
)
from skimage.filters import threshold_otsu
from flyseg.utils.flySeg_reader import file_read, data_save
from flyseg.utils.GMM_HMRF_Body import mask_save, Body_processor,GMM_HMRF_body
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Tuple, List
import csv

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
    alpha=2.0
) -> Tuple[str, str, float, Tuple[int, int, int], str, List[float], List[float], List[float]]:
    """
    Process a single image and return extended results.
    """
    processed_image, threshold, cropped, bbox = process_image(image_path)
    image_shape = tuple(processed_image.shape)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    image_filename = f"{base_filename}.nii.gz"
    image_save_path = os.path.join(save_image_dir, image_filename)

    # Segment
    segmenter = GMM_HMRF_body(n_components=3, alpha=alpha, max_iter=25, neighborhood=26)
    mask, means, covs, weights = segmenter.gmm_fit(cropped)
    # print(mask.shape)
    # Post-process mask and restore
    post_mask = Body_processor.post_mask(mask)
    restored = Body_processor.restore_mask(image_shape, post_mask, bbox)

    # Save
    Body_filename = f"{base_filename}_bodyMask.nii.gz"
    Body_path = os.path.join(body_dir, Body_filename)
    mask_save(restored, body_dir, Body_filename)
    data_save(processed_image, save_image_dir, image_filename, file_format='nii')

    return (
        image_path, image_save_path, threshold, image_shape, Body_path,
        means.flatten().tolist(),
        covs.flatten().tolist(),
        weights.flatten().tolist()
    )

def process_images_multithreaded(input_folder: str, save_image_dir: str, body_dir: str, alpha: float = 2.0, max_workers: int = 3) -> List[Tuple]:
    file_info = []

    h5_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".h5"):
                h5_files.append(os.path.join(root, file))

    total_files = len(h5_files)
    print(f"🧾 Found {total_files} raw .h5 images in '{input_folder}'")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_and_save_image, image_path, save_image_dir, body_dir, alpha): image_path for image_path in h5_files}
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

    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Info", "Original Path", "Processed Path", "Threshold",
            "Shape (Z,Y,X)", "Body Path",
            "Means", "Covariances", "Weights"
        ])

        for entry in file_info:
            try:
                (
                    orig_path, proc_path, threshold, shape, body_path,
                    means, covariances, weights
                ) = entry

                writer.writerow([
                    info_note,
                    orig_path,
                    proc_path,
                    threshold,
                    str(shape),
                    body_path,
                    means,
                    covariances,
                    weights,
                ])
            except Exception as e:
                print(f"⚠️ Failed to write entry to CSV: {e}")


