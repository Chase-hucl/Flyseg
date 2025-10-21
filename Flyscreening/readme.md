# 🧬 FlyDrugScreening

This repository presents **FlyDrugScreening**, a 3D convolutional neural network (3D-CNN)–based framework designed to quantify **tumor morphology** and evaluate **drug efficacy** in a *Drosophila* model of **pancreatic ductal adenocarcinoma (PDAC)**.  

It performs ensemble-based 3D CNN inference using 23 trained models (MC3-18 architecture) to evaluate the probability of classness to cancer morphology in 3D fluorescence or microscopy data. Lower ensemble scores indicate effective drug response (tumor regression), whereas higher scores indicate persistent or progressive tumor states.

---

## 📁 Folder Structure

```
FlyDrugscreening/
├── DrugScreening.py             # Main inference script
├── requirements.txt             # Python dependency list
├── README.md                    # Documentation file
├── samples/                     # Folder for test sample volumes (.h5)
│   ├── sample_001.h5
│   ├── sample_002.h5
│   └── ...
└── models/                      # Folder for trained model 
    ├── best_model_for_date_20231114.tar
    ├── best_model_for_date_20231120.tar
    ├── ...
    └── best_model_for_date_20241017.tar
```

---

## 🧩 Input Data Preparation

All **test sample images** must be stored inside a folder named **`samples/`**, located in the same directory as `main.py`.

### Requirements for input files:
- **File type:** `.h5` (HDF5)
- **Dataset name:** `dataset_1` (3D volume data)

Each `.h5` file is treated as a single 3D test sample for inference.
---

## 🧠 Models Folder

All trained model are stored in **`models/`**.

### Key Details:
- Contains **23 trained 3D-CNN models**, each saved as a `.tar` file:
- The program loads and evaluates all models sequentially, then computes a **weighted mean** of their predicted probabilities.
---



### 📈 Output Results

Example output file `test_result.csv`:
```csv
File Name,Probability of Classness to Cancer Morphology
samples/sample_001.h5,0.9321
samples/sample_002.h5,0.1485
```

---

## 🧰 Installation

Clone this repository and install all dependencies using the included `requirements.txt` file.

```bash
git clone https://github.com/<your-username>/FlyDrugScreening.git
cd FlyDrugScreening
pip install --index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
```

## ▶️ Running the Program

From the project root directory:
```bash
python DrugScreening.py
```