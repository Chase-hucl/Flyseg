# FlySeg 🧬🍃

**FlySeg** is a pretrained, modular image segmentation pipeline tailored for *Drosophila* tissue segmentation using the [nnUNet v2](https://github.com/MIC-DKFZ/nnUNet) framework.

This tool allows researchers to preprocess `.h5` image data, apply trained nnUNet models, and generate postprocessed `.nii.gz` segmentations with label analysis — all in a streamlined and reproducible workflow.

---

## ✨ Features

- ✅ End-to-end nnUNet-based inference pipeline
- ✅ Preprocessing from `.h5` → `.nii.gz` (Otsu + Gaussian smoothing + 3D masking)
- ✅ Prediction via nnUNet v2 with automatic environment setup
- ✅ Postprocessing with statistical label summaries
- ✅ High-performance multi-threaded image processing
- ✅ Command-line interface with flexible arguments

---

## 📦 Installation

1. **Clone the repository:**
```
git clone https://github.com/Chase-hucl/FlySeg.git
pip install -r requirements.txt
python download_model.py

```


## 🚀 Usage Example
```
python -m flyseg.prediction --input "T:\Chenglang\classification\annotation\dataset\20250415\20250415\Control\Good images" --output "T:\Chenglang\test" --application Toxicology --info PMMA_control --date 20250415 --organ CNS

```


## Output data structure (Example)

```text
output_dir/
└── Toxicology/
    └── 20250419/
        └── PMMA_control/
            ├── imagesTs/               # Prepared input for nnUNet
            ├── temporary/             # Raw predictions
            ├── temporary_PP/          # Postprocessed predictions
            ├── prediction_mask/             # Final labeled masks
            ├── label_stats.csv         # Label analysis output
            └── preprocessing_summary.csv

