import numpy as np
from scipy.ndimage import label
import os
import csv
import nibabel as nib
from tqdm import tqdm
from FlySeg.src import file_read
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

process_lock = multiprocessing.Lock()

def process_file(file_path):
    """
    处理单个文件并返回结果。

    参数：
        file_path (str): 文件的完整路径。

    返回：
        dict: 包含文件的分析结果。
    """
    filename = os.path.basename(file_path)
    try:
        # 加载label图像
        label_image = file_read(file_path)  # 根据实际文件格式修改加载方法
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return None

    try:
        # 获取唯一的label值
        unique_labels = np.unique(label_image)
        file_result = {'filename': filename}

        for lbl in unique_labels:
            if lbl == 0:
                continue  # 跳过背景

            # 找到当前label的mask
            mask = label_image == lbl

            # 获取连通区域和数量
            labeled_regions, num_features = label(mask)

            # 计算每个区域的大小
            region_sizes = [(labeled_regions == i).sum() for i in range(1, num_features + 1)]

            # 保存数据
            file_result[f'label_{int(lbl)}'] = {
                'num_features': num_features,
                'region_sizes': region_sizes
            }

        return file_result

    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        return None

def analyze_label_folder(folder_path, output_csv):
    """
    遍历文件夹中的所有label文件，统计每个文件的label_value及对应的num_features和region_sizes。

    参数：
        folder_path (str): 包含label文件的文件夹路径。
        output_csv (str): 输出的CSV文件路径。

    返回：
        None
    """
    # 获取所有文件
    file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.nii.gz')]

    if not file_list:
        print("No .nii.gz files found in the folder.")
        return

    # 初始化结果存储列表
    results = []

    # 使用多线程和tqdm进度条
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file, file_path): file_path for file_path in file_list}
        for future in tqdm(futures, desc="Processing files", total=len(file_list)):
            result = future.result()
            if result:
                results.append(result)

    if not results:
        print("No valid results to write.")
        return

    try:
        # 获取所有可能的label列
        all_labels = sorted({int(key.split('_')[1]) for result in results for key in result.keys() if key.startswith('label_')})

        # 写入CSV文件内容
        with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # 写入表头
            header = ['filename'] + [f'label_{lbl}' for lbl in all_labels]
            writer.writerow(header)

            # 写入每个文件的数据
            for result in results:
                row = [result['filename']]
                for lbl in all_labels:
                    label_key = f'label_{lbl}'
                    if label_key in result:
                        num_features = result[label_key]['num_features']
                        region_sizes = result[label_key]['region_sizes']
                        row.append(f"{num_features}:{region_sizes}")
                    else:
                        row.append("")
                writer.writerow(row)

    except Exception as e:
        print(f"Error writing to CSV file {output_csv}: {e}")

def process_label(label_image, initial_remove_label=20):
    """
    后处理单个label文件。

    参数：
        label_image (ndarray): 3D label图像，包含各个区域的编号。
        initial_remove_label (int): 初始用于标记多余区域的编号，默认为100。

    返回：
        ndarray: 处理后的3D label图像。
    """
    processed_image = np.copy(label_image)
    unique_labels = np.unique(processed_image)
    current_remove_label = initial_remove_label

    # 遍历每个标签（跳过背景0）
    for lbl in unique_labels:
        if lbl == 0:
            continue

        # 获取当前label的mask
        mask = processed_image == lbl

        # 找到mask中的所有连通区域
        labeled_regions, num_features = label(mask)

        if num_features > 1:
            # 如果不连续，定义最大的区域为主label
            region_sizes = [(labeled_regions == i).sum() for i in range(1, num_features + 1)]
            largest_region_idx = np.argmax(region_sizes) + 1

            # 标记最大的区域
            main_region_mask = labeled_regions == largest_region_idx

            # 处理其余区域
            for i in range(1, num_features + 1):
                if i == largest_region_idx:
                    continue

                secondary_region_mask = labeled_regions == i
                # 将其标记为当前的remove_label
                processed_image[secondary_region_mask] = current_remove_label
                current_remove_label += 1

    # 遍历所有独立的remove_label区域进行处理
    for remove_lbl in range(initial_remove_label, current_remove_label):
        remove_mask = processed_image == remove_lbl

        # 遍历remove_label区域并检查相邻标签
        if np.any(remove_mask):
            neighboring_labels = np.unique(processed_image[np.logical_and(~remove_mask,
                                                                          np.logical_or.reduce([
                                                                              np.roll(remove_mask, shift, axis)
                                                                              for axis in range(3) for shift in
                                                                              (-1, 1)]))])
            neighboring_labels = neighboring_labels[neighboring_labels != 0]

            if len(neighboring_labels) == 1:
                # 如果只与一个其它label相邻，则重新赋值为相邻label编号
                processed_image[remove_mask] = neighboring_labels[0]
            else:
                # 否则移除该区域
                processed_image[remove_mask] = 0

    return processed_image

def process_folder(input_folder, output_folder, remove_label=20):
    """
    对整个文件夹中的nii文件进行处理，并保存结果。

    参数：
        input_folder (str): 输入文件夹路径。
        output_folder (str): 输出文件夹路径。
        target_labels (list): 预期的标签编号列表。
        remove_label (int): 用于标记多余区域的编号，默认为100。

    返回：
        None
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_list = [f for f in os.listdir(input_folder) if f.endswith('.nii.gz')]

    def process_and_save(file_name):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        with process_lock:  # 使用进程锁保护
            try:
                # 加载label图像
                label_image = nib.load(input_path).get_fdata().astype(np.int8)

                # 处理label图像
                processed_image = process_label(label_image, remove_label)

                # 保存处理后的图像
                processed_nii = nib.Nifti1Image(processed_image, affine=np.eye(4))
                nib.save(processed_nii, output_path)
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

    with ThreadPoolExecutor(max_workers=os.cpu_count()//2) as executor:  # 限制最大线程数为4
        list(tqdm(executor.map(process_and_save, file_list), total=len(file_list), desc="Processing Labels"))