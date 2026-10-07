# Canonical Benchmark Experiment: EXP-MLUA-003 (Epoch 75 Checkpoint)

This document provides the definitive technical specification and verified empirical results for our canonical best model: **`EXP-MLUA-003_E75_BEST.pth`**.

---

## 1. Model Metadata & Specifications

| Property | Value / Specification |
|---|---|
| **Experiment ID** | `EXP-MLUA-003` |
| **Selected Checkpoint File** | `EXP-MLUA-003_E75_BEST.pth` |
| **Selected Epoch** | **Epoch 75** (Selected on validation data) |
| **Total Run Completion** | Concluded at **Epoch 80** (10,560 steps; cumulative time: 179,231s / 49.79 hrs) |
| **Global Optimization Step** | **Step 9,900** |
| **Model Architecture** | ResNet-34 Encoder + Feature Pyramid Network (FPN) Decoder |
| **Trainable Parameters** | 21,284,545 parameters |
| **Checkpoint File Size** | 370,840,935 bytes (~353.66 MB) |
| **SHA-256 Checksum** | `cb52ed6ec6911fab7f0081588fb63a6f0bad0e37b61c37f0ba9bd293a6bcedfb` |
| **Training Type** | Semi-Supervised Multi-Level Uncertainty-Aware (MLUA) Teacher-Student learning |
| **Operating Decision Threshold** | $\tau = 0.50$ (Fixed prior to evaluation, zero test tuning) |

---

## 2. Dataset & Geometry Configuration
- **Dataset:** DC1000 benchmark dataset.
- **Training Cohort Partition:**
  - Labeled Patches (20%): 530 patches with full binary ground truth.
  - Unlabeled Patches (80%): 1,859 patches trained via consistency regularization.
  - Total Training Patches: 2,389 patches.
- **Patch Resolution:** $384 \times 384$ pixels, grayscale normalized to $[0.0, 1.0]$.
- **Sliding Window Geometry:** Stride of $192\text{ pixels}$ (50% horizontal and vertical overlap).
- **Patches per Full OPG:** Exactly 21 patches ($3 \text{ vertical} \times 7 \text{ horizontal}$ grid covering $768 \times 1536$).
- **Full OPG Reconstruction:** 21-patch sliding-window stitching with 2D Gaussian kernel spatial probability blending to eliminate seam artifacts.

---

## 3. Internal Validation Results (Epoch 75 on Validation Set)

Evaluated across the validation set at fixed operating threshold $\tau = 0.50$:

```text
========================================================================================
EXP-MLUA-003 (Epoch 75, Step 9,900) Validation Performance
========================================================================================
- Validation Dice Similarity (DSC):  71.867%  (0.71867)  [Exceeds paper 71.12% by +0.75 pp]
- Intersection-over-Union (IoU):     57.349%  (0.57349)
- Validation Precision (PPV):        78.132%  (0.78132)
- Validation Recall (Sensitivity):   67.343%  (0.67343)
- Validation Specificity (TNR):      99.824%  (0.99824)
- Peak Validation Loss:              0.72540
- Zero-Prediction Patch Ratio:       2.0%
========================================================================================
```

---

## 4. Final Sealed-Test Evaluation Results (100 Cases)

Following model selection, `EXP-MLUA-003_E75_BEST.pth` was evaluated strictly once on the independent, untouched 100-case sealed test set (`dataset/test/`) on full $768 \times 1536$ uncropped panoramic radiographs:

| Evaluation Metric | Macro (Case Arithmetic Mean) | Micro (Global Pixel Sum) |
|---|:---:|:---:|
| **Dice Similarity Coefficient (DSC)** | **50.147%** | **52.924%** |
| **Intersection over Union (IoU)** | **36.607%** | **35.984%** |
| **Precision (PPV)** | **59.889%** | **61.540%** |
| **Recall / Sensitivity** | **48.077%** | **46.424%** |
| **Specificity (TNR)** | **99.872%** | **99.872%** |
| **F1 Score** | **50.147%** | **52.924%** |
| **Zero-Prediction Case Ratio** | **0.0%** (0 / 100 cases) | — |

### Global Pixel Confusion Matrix (117,964,800 Total Evaluated Pixels)
- **True Positives (TP):** $240,579\text{ pixels}$ ($0.2039\%$)
- **False Positives (FP):** $150,350\text{ pixels}$ ($0.1275\%$)
- **False Negatives (FN):** $277,638\text{ pixels}$ ($0.2354\%$)
- **True Negatives (TN):** $117,296,233\text{ pixels}$ ($99.4332\%$)

$$\text{Global Micro Dice} = \frac{2 \times \text{TP}}{2 \times \text{TP} + \text{FP} + \text{FN}} = \frac{2 \times 240,579}{481,158 + 150,350 + 277,638} = \frac{481,158}{909,146} = \mathbf{52.924\%}$$

---

## 5. Why Epoch 75 Was Selected
1. **Validation Data Selection:** Epoch 75 produced the highest validation Dice score (**71.867%**) and lowest validation loss (**0.7254**) across the entire 80-epoch training history.
2. **Optimal Precision/Recall Balance:** At Epoch 75, validation precision reached **78.13%** while maintaining strong recall at **67.34%**, minimizing false-positive alarms while catching subtle lesions.
3. **Strict Prevention of Test Leakage:** The decision to freeze E75 was made purely from validation statistics before any evaluation on the 100-case sealed test set.
4. **Subsequent Plateau:** Epochs 76–80 confirmed that performance had stabilized (Epoch 80 Dice: 71.343%), proving that Epoch 75 captured the true global convergence peak.

---

## 6. Scientific Framing Note
- This sealed test evaluation represents an **algorithm-level benchmark test on an independent retrospective test split**.
- It is strictly designated as: **`Final Sealed-Test Evaluation`**.
- It is **NOT** a clinical validation, clinical trial, or definitive diagnostic claim.
