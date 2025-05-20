import os
import subprocess

def get_dataset_directory(organ: str, nnunet_raw: str) -> str:
    """
    Map organ name to its corresponding dataset directory under nnUNet_raw.

    Args:
        organ (str): Organ name, like 'CNS', 'Liver', etc.
        nnunet_raw (str): Base path of nnUNet_raw directory.

    Returns:
        str: Full path to the dataset directory.
    """
    organ_map = {
        "CNS": "Dataset206_drl",
        "Nubbin": "Dataset100_nubbin",
        # "Fly": "Dataset203_flygut",
        # "Brain": "Dataset208_brain",
        # Add more mappings here as needed
    }

    if organ not in organ_map:
        raise ValueError(f"❌ Organ '{organ}' not recognized. Please check the available datasets.")

    return os.path.join(nnunet_raw, organ_map[organ])

import os

def get_postprocessing_pkl_path(organ: str, nnunet_results: str) -> str:
    """
    Given an organ name and nnUNet results path, return full path to postprocessing.pkl.

    Args:
        organ (str): Organ name like 'CNS'
        nnunet_results (str): Base path to nnUNet_results

    Returns:
        str: Full path to postprocessing.pkl
    """
    organ_map = {
        "CNS": "Dataset206_drl",
        "Nubbin": "Dataset306_nubbin",
        # "Fly": "Dataset203_flygut",
        # Add more organ-dataset mappings as needed
    }

    if organ not in organ_map:
        raise ValueError(f"Unknown organ '{organ}'. Please check organ_map.")

    dataset_dir = organ_map[organ]
    sub_path = os.path.join(
        dataset_dir,
        "nnUNetTrainer__nnUNetPlans__3d_fullres",
        "crossval_results_folds_0_1_2_3_4",
        "postprocessing.pkl"
    )
    sub_path2 = os.path.join(
        dataset_dir,
        "nnUNetTrainer__nnUNetPlans__3d_fullres",
        "crossval_results_folds_0_1_2_3_4",
        "plans.json"
    )

    full_path = os.path.join(nnunet_results, sub_path)
    full_path2 = os.path.join(nnunet_results,sub_path2)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"❌ postprocessing.pkl not found: {full_path}")
    if not os.path.isfile(full_path2):
        raise FileNotFoundError(f"❌ postprocessing.pkl not found: {full_path2}")

    return full_path, full_path2


def run_nnUNet_prediction(input_folder, output_folder, organ="CNS", num_parts=None):
    """
    Runs nnUNetv2 prediction using organ name or dataset ID.

    Args:
        input_folder (str): Path to the input images (e.g. imagesTs).
        output_folder (str): Path where the predictions will be saved.
        organ (str or int): Organ name or dataset ID (e.g., "CNS" or 206).
        num_parts (str, optional): Specific folds to use (e.g., "0 1 2 3 4").
    """
    organ_to_dataset = {
        "CNS": "Dataset206_drl",  # ✅ 你想要的映射
        "Nubbin": "Dataset306_nubbin"
    }
    if isinstance(organ, int) or (isinstance(organ, str) and organ.isdigit()):
        dataset_id = str(organ)
    elif organ in organ_to_dataset:
        dataset_id = organ_to_dataset[organ]
    else:
        raise ValueError(f"❌ Unsupported organ type: '{organ}'. Please check your input.")

    command = [
        "nnUNetv2_predict",
        "-i", input_folder,
        "-o", output_folder,
        "-d", dataset_id,
        "-tr", "nnUNetTrainer",
        "-c", "3d_fullres",
        "-p", "nnUNetPlans"
    ]

    if num_parts:
        command.extend(["-f"] + num_parts.split())

    print("🧠 Running nnUNet prediction with:")
    print("    " + " ".join(command))

    try:
        subprocess.run(command, check=True)
        print(f"✅ Prediction completed for {organ} ({dataset_id})")
    except subprocess.CalledProcessError as e:
        print(f"❌ Prediction failed: {e}")



def apply_postprocessing(input_dir, output_dir, pp_pkl_file, np=8, plans_json=None):
    """
    Applies nnUNetv2 postprocessing to the predicted segmentations.

    Args:
        input_dir (str): Directory containing predicted segmentations.
        output_dir (str): Directory to save postprocessed results.
        pp_pkl_file (str): Path to postprocessing.pkl file.
        np (int, optional): Number of threads to use. Defaults to 8.
        plans_json (str, optional): Path to plans.json (if required).
    """
    command = [
        "nnUNetv2_apply_postprocessing",
        "-i", input_dir,
        "-o", output_dir,
        "-pp_pkl_file", pp_pkl_file,
        "-np", str(np)
    ]

    if plans_json:
        command.extend(["-plans_json", plans_json])

    print("🛠 Running postprocessing:")
    print("    " + " ".join(command))

    os.makedirs(output_dir, exist_ok=True)

    try:
        subprocess.run(command, check=True)
        print("✅ Postprocessing completed.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Postprocessing failed: {e}")
