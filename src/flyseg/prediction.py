import argparse
import os
from flyseg.nnunet_config import configure_nnunet_environment
from flyseg.nnunet_runner import run_nnUNet_prediction, apply_postprocessing,get_dataset_directory,get_postprocessing_pkl_path
from flyseg.preprocessor import process_images_multithreaded
from flyseg.preprocessor import export_file_info_to_csv
from flyseg.utils.folder_cleaner import clear_folder
from flyseg.utils.data_dispatcher import copy_and_rename_files_multithreaded
from flyseg.postprocessor import analyze_label_folder, process_folder
import sys
import torch



def main():
    print("CUDA available:", torch.cuda.is_available())
    parser = argparse.ArgumentParser(description="FlySeg nnUNet prediction pipeline")

    # 输入输出相关参数
    parser.add_argument("--input", "-i", required=True, help="Path to input raw image folder (.h5)")
    parser.add_argument("--output", "-o", required=True, help="Path to output all prediction image files folder (.nii file)")
    parser.add_argument("--application", type = str, default="Toxicology",
                        help="the first-class folder in the output")
    parser.add_argument("--info","-info", required=True,help= "set the experiment infomation, e.g. -PMMA_control")
    parser.add_argument("--date", "-date", required=True, help="set the experiment date, e.g. 20250418")
    # parser.add_argument("--output", "-o", required=True, help="Path to save final prediction results")
    parser.add_argument("--organ", type=str, required=True, help="please input Organ name (e.g., CNS or nubbin)")
    parser.add_argument("--folds", type=str, default="0 1 2 3 4", help="Folds to use for prediction")

    # 后处理参数
    # parser.add_argument("--pp_pkl", type=str, required=True, help="Path to postprocessing.pkl")
    parser.add_argument("--plans_json", type=str, help="Optional path to plans.json")
    parser.add_argument("--np", type=int, default=8, help="Number of threads for postprocessing")

    args = parser.parse_args()

    print("🧪 Step 0: Configuring environment")
    nnunet_raw, nnunet_results= configure_nnunet_environment()

    if not os.path.exists(args.input):
        print(f"Error in find raw images folder {args.input}")
        return
    if not os.path.exists(args.output):
        os.makedirs(args.output,exist_ok=True)
    print("🔄 Step 1: Preprocessing input images")
    output_1stdir = os.path.join(args.output, f"{args.application}")
    output_dir = os.path.join(output_1stdir,f"{args.date}")
    output_2eddir = os.path.join(output_dir,f"{args.info}")
    imagesTs_dir = os.path.join(output_2eddir, "imagesTs")
    bodyMask_dir = os.path.join(output_2eddir, "Body_mask")
    labelTs_dir = os.path.join(output_2eddir, "temporary_mask")
    post_dir = os.path.join(output_2eddir, "temporary_mask_PP")
    pp_dir = os.path.join(output_2eddir,"prediction_mask")
    os.makedirs(output_1stdir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_2eddir, exist_ok=True)
    os.makedirs(imagesTs_dir, exist_ok=True)
    os.makedirs(bodyMask_dir, exist_ok=True)
    os.makedirs(labelTs_dir, exist_ok=True)
    os.makedirs(post_dir, exist_ok=True)
    os.makedirs(pp_dir,exist_ok=True)

    file_info = process_images_multithreaded(args.input, imagesTs_dir)
    csv_path = os.path.join(output_2eddir, "preprocessing_summary.csv")
    export_file_info_to_csv(file_info, csv_path, info_note= args.info)

    print("🤖 Step 2: Running nnUNet prediction")
    direction = get_dataset_directory(args.organ, nnunet_raw)
    prediction_dir = os.path.join(direction, "imagesTs")
    clear_folder(prediction_dir,remove_subdirs=True, verbose=False)
    copy_and_rename_files_multithreaded(csv_path,prediction_dir)
    run_nnUNet_prediction(prediction_dir, labelTs_dir, organ=args.organ, num_parts=args.folds)
    #
    print("🧹 Step 3: Running postprocessing")
    pkl_file_path, plan_json= get_postprocessing_pkl_path(args.organ,nnunet_results)
    apply_postprocessing(labelTs_dir, post_dir, pp_pkl_file=pkl_file_path, plans_json=plan_json, np=args.np)
    #
    print("📊 Step 4: Analyzing final results")
    label_states_csv = os.path.join(pp_dir, "label_stats.csv")
    process_folder(post_dir,pp_dir)
    analyze_label_folder(pp_dir,label_states_csv)
    #
    print("✅ Pipeline finished successfully!")
    #

if __name__ == "__main__":
    # 测试时注入参数（仿真命令行调用）
    if len(sys.argv) == 1:
        sys.argv += [
            "--input", r"T:\Chenglang\classification\annotation\dataset\20250415\20250415\Control\Good images\test",
            "--output", r"T:/Chenglang/test_output",
            "--application", "Toxicology",
            "--info", "PMMA_control",
            "--date", "20250415",
            "--organ", "CNS"
        ]
    main()
