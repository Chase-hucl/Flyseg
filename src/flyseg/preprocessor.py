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
    subset = image[(image > lower_bound) & (image < upper_bound)]
    threshold = threshold_otsu(subset) if subset.size > 0 else lower_bound
    thresholded_image = np.where(image >= threshold, image, 0)
    return thresholded_image, threshold


def process_image(image_path: str, sigma: float = 2.0) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Processes a 3D image: thresholding, smoothing, and extracting largest component.

    Args:
        image_path: Path to input image.
        sigma: Standard deviation for Gaussian filter.

    Returns:
        Tuple of processed image, threshold, and binary mask.
    """
    image = file_read(image_path).astype(np.uint16)
    thresholded_image, threshold = apply_otsu(image, 20, 1000)
    smoothed_image = gaussian_filter(thresholded_image, sigma=sigma)
    binary_mask = smoothed_image > 0
    labeled_image, _ = label(binary_mask)
    component_sizes = np.bincount(labeled_image.ravel())
    largest_component_label = component_sizes[1:].argmax() + 1
    largest_component_mask = labeled_image == largest_component_label
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
    else:
        processed_image = image

    return processed_image, threshold, final_mask


def process_and_save_image(
    image_path: str,
    save_image_dir: str,
) -> Tuple[str, float, Tuple[int, int, int]]:
    """
    Process a single image and save the processed result and mask.

    Returns:
        original image path, saved image path, threshold, shape, mask path
    """
    processed_image, threshold, _ = process_image(image_path)
    image_shape = tuple(processed_image.shape)

    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    image_filename = f"{base_filename}.nii.gz"
    # mask_filename = f'FINDS_{index:04d}_mask.nii.gz'

    image_save_path = os.path.join(save_image_dir, image_filename)
    # mask_save_path = os.path.join(save_mask_dir, mask_filename)

    data_save(processed_image, save_image_dir, image_filename, file_format='nii')
    # sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    # sitk.WriteImage(sitk_mask, mask_save_path)

    return image_path, image_save_path, threshold, image_shape
            # , mask_save_path)

def process_images_multithreaded(
    input_folder: str,
    save_image_dir: str,
    # save_mask_dir: str
) -> List[Tuple[str, str, float, Tuple[int, int, int]]]:
    """
    Multi-threaded processing of all .h5 images in a directory.

    Args:
        input_folder: Root directory containing .h5 files.
        save_image_dir: Directory to save processed images.
        save_mask_dir: Directory to save masks.

    Returns:
        List of processing summaries per image.
    """
    file_info = []
    num_workers = os.cpu_count()//2
    futures = []

    h5_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".h5"):
                h5_files.append(os.path.join(root, file))

    total_files = len(h5_files)
    print(f"🧾 Found {total_files} raw .h5 images in '{input_folder}'")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        with tqdm(total=total_files, desc="Processing images") as pbar:
            for image_path in h5_files:
                futures.append(
                    executor.submit(process_and_save_image, image_path, save_image_dir)
                )
            for future in as_completed(futures):
                try:
                    result = future.result()
                    file_info.append(result)
                except Exception as e:
                    print(f"❌ Failed to process image: {e}")
                pbar.update(1)
    return file_info

def export_file_info_to_csv(
    file_info: List[Tuple[str, str, float, Tuple[int, int, int]]],
    csv_path: str,
    info_note: str = ""
) -> None:
    """
    Export processed image info to a CSV file.

    If the file exists, data will be appended. The first line will include an optional info note as a comment.

    Args:
        file_info: List of tuples containing (original_path, saved_path, threshold, shape)
        csv_path: Path to the output CSV file
        info_note: Optional string to add as a header comment (e.g., "CNS_20250419")
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header if new file
        writer.writerow(["Info","Original Path", "Processed Path", "Threshold", "Shape (Z,Y,X)"])

        # Write data
        for entry in file_info:
            orig, saved, threshold, shape = entry
            writer.writerow([info_note,orig, saved, threshold, str(shape)])
