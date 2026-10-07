# DC1000 Dataset Documentation & Provenance

## 1. Dataset Name
**DC1000** (Dental Caries 1,000 Panoramic Radiographs Dataset).

---

## 2. Source & Clinical Origin
- **Originating Institution:** Department of Stomatology, Zhejiang Provincial People's Hospital (Hangzhou, China).
- **Primary Literature Reference:** Xianyun Wang, Sizhe Gao, Kaisheng Jiang, Huicong Zhang, Linhong Wang, Feng Chen, Jun Yu, Fan Yang, *"Multi-level uncertainty aware learning for semi-supervised dental panoramic caries segmentation"*, *Neurocomputing*, Volume 540, 2023, Article 126208. DOI: [10.1016/j.neucom.2023.03.069](https://doi.org/10.1016/j.neucom.2023.03.069).

---

## 3. Total Images & Cohort Breakdown

### Paper-Reported Dataset Facts
- **Total Panoramic Radiographs (OPGs):** 1,000 full-mouth orthopantomograms.
- **Detailed Consensus Annotations:** 593 cases (containing 4,520 clinically verified caries lesions; mean 7.6 lesions per OPG).
- **Rough / Ambiguous Annotations:** 407 cases (incorporated as unlabeled cohort for semi-supervised learning).
- **Annotators:** 5 licensed clinical dental practitioners with double-blind review and consensus adjudication.

### Current Recovered Repository / Archive Facts
- **Raw Dataset Archive:** `data/raw/DC1000_dataset.zip` (1,099,499,277 bytes; ~1.02 GB uncompressed).
- **Training Cohort Directory (`data/raw/DC1000_dataset/train/`):**
  - Images (`images/`): 900 panoramic OPG radiographs (`.png`).
  - Labels (`labels/`): 900 corresponding ground-truth segmentation masks (`.png`).
- **Sealed Test Set Directory (`dataset/test/`):**
  - Images (`images/`): 100 panoramic OPG radiographs (`.png`).
  - Labels (`labels/`): 100 corresponding ground-truth segmentation masks (`.png`).
  - Cut Labels (`labels_cut/`): 100 dental-arch cropped label masks (`.png`).
- **Total Radiographs Recovered in Repository:** 900 training + 100 sealed test = **1,000 panoramic radiographs**.

---

## 4. Annotation Information
- **Annotation Format:** Single-channel 8-bit binary PNG masks ($0 = \text{background / sound tissue}$, $255 = \text{caries lesion}$).
- **Target Category:** Pixel-level binary semantic segmentation of demineralized tooth structure (dental caries).
- **Clinical Depth Annotations (Base Paper):**
  - Shallow caries ($DICE_s$): Mean lesion area $\approx 160\text{ pixels}$ ($< 300\text{ px}$).
  - Middle caries ($DICE_m$): Mean lesion area $\approx 474\text{ pixels}$ ($300 - 1000\text{ px}$).
  - Deep caries ($DICE_l$): Mean lesion area $\approx 1,952\text{ pixels}$ ($> 1000\text{ px}$).

---

## 5. Training / Validation / Test Partition Information

### In Our Implementation (`EXP-MLUA-003`):
- **Raw Image Dimensions:** $2943 \times 1435$ pixels (standardized canvas: $768 \times 1536$ pixels preserving 1:2 anatomical aspect ratio).
- **Training Patch Geometry:** $384 \times 384$ pixels extracted with horizontal and vertical stride of $192\text{ pixels}$ (yielding 21 patches per panoramic image).
- **Training Cohort Split:**
  - Total Training Patches: 2,389 patches.
  - Labeled Training Patches (20%): 530 patches.
  - Unlabeled Training Patches (80%): 1,859 patches.
- **Validation Set:** 50 patches held out for epoch-by-epoch checkpoint validation.
- **Independent Sealed Test Set:** Exactly 100 full-mouth panoramic radiographs ($768 \times 1536$), evaluated once using 21-patch sliding-window inference with 2D Gaussian spatial probability blending at $\tau = 0.50$. Zero test cases were used for training or threshold tuning.

---

## 6. Image & Mask Format
- **Radiograph Format:** 8-bit grayscale PNG images ($1 \times H \times W$).
- **Mask Format:** 8-bit binary PNG ($0 = \text{sound dental tissue / bone / air}$, $255 = \text{carious lesion}$).
- **Normalized Model Tensor:** Values scaled to $[0.0, 1.0]$ float32.

---

## 7. Preprocessing Used in Our Project
1. **Aspect-Preserving Canvas Resizing:** Standardized to $768 \times 1536$ pixels (1:2 aspect ratio).
2. **CLAHE Enhancement:** Contrast Limited Adaptive Histogram Equalization applied to normalize multi-scanner sensor variability and dental contrast.
3. **Sliding-Window Patch Decomposition:** $384 \times 384$ pixel patches extracted at $192\text{ px}$ stride (21 patches per full OPG).
4. **Data Augmentation (Training Only):** Random horizontal flipping, random rotation ($\pm 10^\circ$), and subtle brightness jittering.

---

## 8. Important Class Imbalance Observation
- **Full Panoramic OPG Imbalance:** On full uncropped images ($768 \times 1536 = 1,179,648\text{ pixels}$), true caries lesions occupy an average of only **$\approx 0.15\%$ (1.5‰)** of pixels. Over $99.85\%$ of pixels are sound enamel, dentin, bone, pulp, restorations, or air background.
- **Patch Amplification Effect:** Cropping into $384 \times 384$ patches with positive lesion centering amplifies foreground lesion concentration to **$11.79\%$** ($\approx 80\times$ density increase), preventing zero-gradient collapse during gradient descent.

---

## 9. Important Limitations
1. **2D Projection Ambiguity:** Overlapping anatomy in panoramic tomography creates cervical burnout shadows that mimic interproximal root caries.
2. **Restoration Confounders:** Radiolucent resin composite fillings exhibit density similar to active demineralization and cannot be definitively separated without restorative treatment history.
3. **Scanner Heterogeneity:** Tube voltage ($kVp$) and sensor exposure variations alter background gray levels across clinics.
4. **Binary Nature of Masks:** Ground truth masks provide spatial boundaries but lack separate voxel-depth channels.

---

## 10. Exact Source of Each Number
- *1,000 images, 593 detailed, 407 rough, 4,520 lesions, 160/474/1952 px lesion sizes:* Base research paper (*Neurocomputing 2023*, Section 4.1 & 5.4).
- *900 train images, 100 test images:* Directory file counts in `data/raw/DC1000_dataset/` and `dataset/test/`.
- *2,389 total patches, 530 labeled (20%), 1,859 unlabeled (80%):* `src/mlua/data/` and `outputs/experiments/EXP-MLUA-003_FINAL/TRAINING_STATUS.md`.
- *768 x 1536 canvas, 384 x 384 patch, 192 stride, 21 patches/OPG:* `src/mlua/models/fpn.py` and `src/mlua/evaluation/final_100_evaluation.py`.
- *117,964,800 evaluated test pixels, TP=240,579, FP=150,350, FN=277,638, TN=117,296,233:* `outputs/diagnostics/EXP-MLUA-003_SEALED_TEST/EXP-MLUA-003_E75_FINAL_TEST_SUMMARY.csv`.
