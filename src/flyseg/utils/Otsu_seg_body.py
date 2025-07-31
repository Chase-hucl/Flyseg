import numpy as np
import os
import nibabel as nib
from scipy.ndimage import (
    gaussian_filter, label, binary_fill_holes, binary_closing, generate_binary_structure
)
from skimage.filters import threshold_otsu
from Data_read import file_read
from Image_view_v3 import Image_viewer_simple
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import traceback

def fill_holes_by_slice(mask):
    """
    对 3D 二值 mask 沿 Z 轴方向逐 slice 填洞，并保留每个 slice 上最大的连通区域。
    同时记录相邻 slice 间的像素差异是否超过 200%。
    前后各10张slice将跳过跳变检测和区域裁剪。
    """
    filled_mask = np.zeros_like(mask, dtype=np.uint8)
    previous_area = None
    needs_check = False

    # 找出每个 slice 是否为非空
    z_sums = [np.sum(mask[:, :, z]) for z in range(mask.shape[2])]
    non_empty_slices = [z for z, s in enumerate(z_sums) if s > 0]

    if len(non_empty_slices) < 21:
        process_range = []  # 少于 21 层，跳过处理
    else:
        process_range = non_empty_slices[5:-5]  # 中间可处理的 slice 区间

    for z in range(mask.shape[2]):
        slice_2d = mask[:, :, z]
        filled_2d = binary_fill_holes(slice_2d).astype(np.uint8)
        labeled, num_features = label(filled_2d)
        #
        # print(f"Slice {z}: connected components = {num_features}")

        if num_features == 0:
            continue
        elif num_features == 1 or z not in process_range:
            largest_region = filled_2d
        else:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            largest_label = np.argmax(sizes)
            largest_region = (labeled == largest_label).astype(np.uint8)

            # 面积跳变检查
            area = int(np.sum(largest_region))
            if previous_area is not None and previous_area > 0:
                change_ratio = abs(area - previous_area) / previous_area
                if change_ratio > 2:
                    needs_check = True
            previous_area = area

        filled_mask[:, :, z] = largest_region

    return filled_mask, needs_check

def clean_filename(filename):
    """
    清除尾缀 _bodymask.nii.gz 和 .dcimg
    """
    name = filename.replace("_bodymask", "")
    name = name.replace(".dcimg", "")
    return name

def data_save(mask, save_dir, filename, dtype=np.uint8):
    """
    保存 3D mask 为 NIfTI 文件 (.nii.gz)
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{filename}.nii.gz")
    nii_img = nib.Nifti1Image(mask.astype(dtype), affine=np.eye(4))
    nib.save(nii_img, save_path)
    print(f"✅ Saved: {save_path}")

def apply_otsu(image, low, high):
    """
    使用 Otsu 方法阈值化图像
    """
    data = image[(image > low) & (image < high)]
    thresh = threshold_otsu(data) if data.size > 0 else low
    image[image < thresh] = 0
    return image

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
    filled_region, needs_check = fill_holes_by_slice(region)
    return filled_region, needs_check

def post_mask(mask):
    """
    进一步平滑并保留最大连通区域
    """
    mask = (mask > 0).astype(np.uint8)
    mask = binary_fill_holes(mask).astype(np.uint8)
    smoothed = gaussian_filter(mask.astype(np.float32), sigma=2)
    smoothed_binary = (smoothed > 0.5).astype(np.uint8)
    filled_mask = binary_fill_holes(smoothed_binary).astype(np.uint8)

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

    mask = remove_largest(filled_mask, 2)
    mask = remove_largest(mask, 2)
    return mask

def process_file(file_path, output_folder, intensity_min, intensity_max):
    try:
        img = file_read(file_path)
        mask1, needs_check = apply_otsu_and_fill(img, intensity_min, intensity_max)
        mask2 = post_mask(mask1)
        voxel_count = int(np.count_nonzero(mask2))
        # print(f"mask2 dtype: {mask2.dtype}, shape: {mask2.shape}, unique values: {np.unique(mask2)}")
        filename = os.path.splitext(os.path.basename(file_path))[0] + "_bodymask"
        data_save(mask2, output_folder, filename)
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

def run_pipeline(input_dir, output_dir, summary_csv_path, max_workers=50, intensity_min=0, intensity_max=300):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
    print(f"🧾 Running with intensity range: [{intensity_min}, {intensity_max}]")
    files = [
        os.path.join(dp, f)
        for dp, _, filenames in os.walk(input_dir)
        for f in filenames if f.endswith(".h5")
    ]
    print(f"🔍 Found {len(files)} .h5 files.")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for f in files:
            rel_path = os.path.relpath(f, input_dir)
            output_subdir = os.path.join(output_dir, os.path.dirname(rel_path))
            os.makedirs(output_subdir, exist_ok=True)

            future = executor.submit(process_file, f, output_subdir, intensity_min, intensity_max)
            futures[future] = f

        for future in tqdm(as_completed(futures), total=len(futures), desc="🧠 Processing"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                err_file = futures[future]
                results.append({
                    "file": err_file,
                    "status": f"exception: {str(e)}",
                    "traceback": traceback.format_exc()
                })

    # 保存结果到 CSV
    # 保存结果到 CSV
    records = []
    for row in results:
        if row["status"] == "success":
            records.append({
                "filename": row["filename"],
                "voxel_count": row["voxel_count"],
                "needs_check": row["needs_check"],
                "status": row["status"]
            })
        else:
            records.append({
                "filename": row.get("filename", "unknown"),
                "status": row["status"],
                "error": row.get("error", "")
            })

    df = pd.DataFrame(records)
    df.to_csv(summary_csv_path, index=False)
    print(f"✅ Finished. Results saved to:\n→ {summary_csv_path}")