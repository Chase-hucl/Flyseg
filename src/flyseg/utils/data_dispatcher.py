import os,re
import pandas as pd
from tqdm import tqdm
from typing import Dict, List
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from flyseg.utils.flySeg_reader import file_read,data_save

def process_row_with_index(index: int, row: pd.Series, destination_folder: str) -> Dict:
    """
    Process a single row: read image, rename to Finds_XXXX.nii.gz, save, and return mapping.
    """
    result = {
        "Index": index,
        "Original Path": None,
        "Renamed Path": None,
        "Status": "Failed"
    }
    try:
        src_path = row['Processed Path']
        # print(f"Processing file: {data_path}")
        if not isinstance(src_path, str) or not os.path.exists(src_path):
            result["Status"] = f"❌ File not found: {src_path}"
            print(result["Status"])
            return result

        # New filename with zero-padded index
        new_filename = f"Finds_{index:04d}_0000.nii.gz"
        dst_path = os.path.join(destination_folder, new_filename)
        shutil.copy(src_path,dst_path)
        # print(f"Saved file to: {save_path}")

        if os.path.exists(dst_path):
            result.update({
                "Renamed Path": dst_path,
                "New Filename": new_filename,
                "Status": "Success"
            })
            # print(f"✅ Copied: {src_path} → {dst_path}")
        else:
            result["Status"] = f"❌ Copied but file not found at destination: {dst_path}"
            print(result["Status"])

    except Exception as e:
        result["Status"] = f"❌ Exception: {str(e)}"
        print(result["Status"])

    return result


import os
import pandas as pd
from pathlib import Path
from natsort import natsorted

def rename_files_from_csv(csv_path: str, folder: str) -> None:
    """
    Rename files based on mappings in a CSV file. The CSV must contain 'Renamed Path' and 'Original Path' columns.

    Rules:
        - 'Renamed Path' filenames must end with '_0000.nii.gz'; this will be replaced with '.nii.gz'
        - 'Original Path' filenames must end with '.dcimg.h5'; this will be replaced with '.nii.gz'
        - Files will be renamed from folder/new_name to folder/old_name

    Parameters:
        csv_path (str): Path to the CSV file.
        folder (str): Base directory containing the files to rename.

    Raises:
        FileNotFoundError: If CSV file or expected source file does not exist.
        ValueError: If filename patterns are invalid.
    """
    csv_file = Path(csv_path)
    root = Path(folder)

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    if not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {root}")

    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        renamed_name = Path(row['Renamed Path']).name
        original_name = Path(row['Original Path']).name

        if not renamed_name.endswith("_0000.nii.gz"):
            continue
        if not original_name.endswith(".dcimg.h5"):
            continue

        new_name = renamed_name.replace("_0000.nii.gz", ".nii.gz")
        old_name = original_name.replace(".dcimg.h5", ".nii.gz")

        src = root / new_name
        dst = root / old_name

        if not src.exists():
            raise FileNotFoundError(f"Source file does not exist: {src}")

        src.rename(dst)


def copy_and_rename_files_multithreaded(
    csv_path: str,
    destination_folder: str,
):
    """
    Copies and renames images based on a CSV file with 'Processed Image Path',
    and writes a mapping CSV for future recovery of original filenames.
    """
    os.makedirs(destination_folder, exist_ok=True)
    df = pd.read_csv(csv_path)
    # print(df)
    # print(f"Read {len(df)} rows from {excel_path}")

    results: List[Dict] = []
    max_worker = os.cpu_count()//2
    with ThreadPoolExecutor(max_workers = max_worker) as executor:
        futures = {
            executor.submit(process_row_with_index, idx, row, destination_folder): idx
            for idx, row in df.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Dispatching images"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"❌ Thread error: {e}")
    # print(results)
    results_df = pd.DataFrame(results).set_index("Index")
    df = pd.concat([df, results_df[["Renamed Path", "Status"]]], axis=1)
    df.to_csv(csv_path, index=False)
    # files = natsorted(os.listdir(destination_folder))

    # print(f"✅ Updated input CSV with renaming info: {csv_path}")

    # Save mapping for output back-rename
    # csv_dir = os.path.dirname(os.path.abspath(csv_path))
    # mapping_output_path = os.path.join(csv_dir, "image_mapping.h5")
    # mapping_df = results_df[["Renamed Path", "Original Path"]].copy()
    # mapping_df.to_hdf(mapping_output_path, key="mapping", mode="w")
    # print(f"📦 Saved mapping HDF5 to: {mapping_output_path}")