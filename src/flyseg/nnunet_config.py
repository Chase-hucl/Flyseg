import os

def is_unc(path: str) -> bool:
    return path.startswith("\\\\") or path.startswith("//")

def configure_nnunet_environment():
    """
    Automatically configures the required nnUNet environment variables based on the
    pretrained model directory structure within the project.

    This function sets the following environment variables:
        - nnUNet_results:        path to trained model results
        - nnUNet_raw:            path to raw input data (imagesTr/imagesTs)
        - nnUNet_preprocessed:   path to preprocessed data

    It assumes the following directory layout:
        project_root/
            pretrained_model/
                nnUNet/
                    nnUNet_results/
                    nnUNet_raw/
                    nnUNet_preprocessed/

    Raises:
        FileNotFoundError: if nnUNet_results directory is not found.
    """

    # Resolve the project root from the current file location
    current_file = os.path.abspath(__file__)
    project_root = os.path.abspath(os.path.join(current_file, "../../.."))
    nnunet_base_path = os.path.join(project_root, "pretrained_model", "nnUNet")

    # Define paths for nnUNet directories
    nnunet_results = os.path.join(nnunet_base_path, "nnUNet_results")
    nnunet_raw = os.path.join(nnunet_base_path, "nnUNet_raw")
    nnunet_preprocessed = os.path.join(nnunet_base_path, "nnUNet_preprocessed")

    # Ensure the model directory exists
    if not os.path.isdir(nnunet_results):
        raise FileNotFoundError(f"❌ nnUNet results directory not found: {nnunet_results}")

    # Set environment variables for nnUNet
    os.environ["nnUNet_results"] = nnunet_results
    os.environ["nnUNet_raw"] = nnunet_raw
    os.environ["nnUNet_preprocessed"] = nnunet_preprocessed
    print(nnunet_raw)
    return nnunet_raw, nnunet_results


    # print("✅ nnUNet environment variables successfully configured:")
    # print(f"  🧠 nnUNet_results      = {nnunet_results}")
    # print(f"  📦 nnUNet_raw          = {nnunet_raw}")
    # print(f"  📦 nnUNet_preprocessed = {nnunet_preprocessed}")

if __name__ == "__main__":
    configure_nnunet_environment()