# EXP-MLUA-003: Final Independent Sealed Test Set Evaluation Report

**Project:** EXP-MLUA-003 (ResNet-34 + FPN MLUA Semi-Supervised Segmentation)  
**Evaluation Status:** COMPLETE & FROZEN  
**Evaluation Type:** Independent Sealed Test Evaluation (Read-Only)  
**Date:** September 12, 2026  

---

## 1. Executive Summary

This report documents the final independent benchmark evaluation for **EXP-MLUA-003** on the sealed 100-case test set (`data/test/` / `dataset/test/`). 

Training was formally concluded and frozen across 60 epochs (7,920 global steps). A rigorous, read-only validation threshold sensitivity analysis established that the canonical operating threshold $\tau = 0.50$ is globally optimal for validation Dice. Consequently, the frozen **Epoch 56 BEST checkpoint** (`EXP-MLUA-003_BEST.pth`) was evaluated on the sealed test set strictly at $\tau = 0.50$ without any test-time tuning, threshold sweeping, or weight modifications.

### Key Benchmark Findings:
- **Test Dataset Size:** 100 full panoramic cases (768×1536 resolution), 100% evaluated.
- **Macro Test Dice:** **`43.041%`** (Macro IoU: `29.057%`, Precision: `41.244%`, Recall: `52.896%`, Specificity: `99.630%`).
- **Micro Test Dice:** **`43.391%`** (Micro IoU: `27.707%`, Precision: `37.795%`, Recall: `50.931%`, Specificity: `99.630%`).
- **Zero-Prediction Ratio:** **`0.0%`** (0 of 100 cases produced empty predictions; the model actively segmented anatomical mandibular canal structures across all test cases).
- **Pixel Statistics:** Total TP = `263,935`, FP = `434,392`, FN = `254,282`, TN = `117,012,191`.

---

## 2. Model and Checkpoint Verification

The exact checkpoint loaded for evaluation was:
- **Path:** [`outputs/experiments/EXP-MLUA-003/checkpoints/EXP-MLUA-003_BEST.pth`](file:///c:/Users/devin/MLUA/outputs/experiments/EXP-MLUA-003/checkpoints/EXP-MLUA-003_BEST.pth)
- **Epoch:** `56`
- **Global Step:** `7392`
- **Architecture:** ResNet-34 Backbone + Feature Pyramid Network (FPN) Multi-Level Uncertainty-Aware (MLUA) Semi-Supervised Model (Auxiliary Deep Supervision Heads at stages 1–4 + Fused Final Segmentation Head).
- **Parameter Health:** 100% finite (0 NaNs, 0 Infs verified across all student & teacher tensors).
- **Preserved Checkpoint Validation Metrics:**
  - Validation Dice: `65.623%`
  - Validation IoU: `49.854%`
  - Validation Precision: `69.009%`
  - Validation Recall: `63.649%`
  - Validation Specificity: `99.753%`
  - Validation Loss: `0.7639`

---

## 3. Validation-Selected Operating Threshold

The operating threshold was fixed prior to test evaluation:
- **Selected Threshold:** $\tau = 0.50$
- **Selection Basis:** Canonical validation set threshold sweep ($\tau \in [0.05, 0.95]$) conducted in `EXP-MLUA-003_THRESHOLD_ANALYSIS`.
- **Integrity Rule:** No threshold tuning, search, or adaptation was performed on the sealed test data. The test set was evaluated exclusively at $\tau = 0.50$.

---

## 4. Test Dataset Size & Verification

- **Test Images Directory:** `dataset/test/images_cut` (100 PNG files)
- **Test Labels Directory:** `dataset/test/labels_cut` (100 PNG files)
- **Case Count:** Exactly 100 sealed cases evaluated.
- **Completeness:** 100/100 cases successfully processed (0 missing, 0 partial, 0 corrupted).

---

## 5. Test Evaluation Protocol

The test inference and panoramic reconstruction protocol identically replicated the validation pipeline:
1. **Resolution & Geometry:** Full panoramic image dimension of $768 \times 1536$ pixels.
2. **Sliding Window Extraction:** 21 overlapping patches of size $384 \times 384$ pixels extracted with a stride of $192$ pixels ($3 \times 7$ grid).
3. **Preprocessing & Normalization:** Channel normalization to $[0, 1]$ floating-point tensor.
4. **Inference Mode:** `torch.inference_mode()` with frozen weights.
5. **Reconstruction:** Overlap-averaged reconstruction accumulating probabilities across intersecting patch coordinates and normalizing by spatial patch overlap counts.
6. **Binarization:** Fixed threshold $\tau = 0.50$ applied to the reconstructed continuous probability map.
7. **Metric Calculation:** Standard macro (unweighted per-case mean) and micro (global pixel sum accumulation) calculation matching existing repo evaluation utilities.

---

## 6. Final Test Metrics

### Global Test Performance Summary

| Metric | Macro (Case Mean) | Micro (Global Pixel Sum) |
| :--- | :---: | :---: |
| **Dice Coefficient** | **`43.041%`** | **`43.391%`** |
| **IoU / Jaccard** | **`29.057%`** | **`27.707%`** |
| **Precision** | **`41.244%`** | **`37.795%`** |
| **Recall (Sensitivity)** | **`52.896%`** | **`50.931%`** |
| **Specificity** | **`99.630%`** | **`99.630%`** |
| **F1 Score** | **`43.041%`** | **`43.391%`** |
| **Zero-Prediction Ratio** | **`0.0%`** (0 / 100 cases) | — |

### Global Pixel Confusion Matrix
- **True Positives (TP):** `263,935` pixels
- **False Positives (FP):** `434,392` pixels
- **False Negatives (FN):** `254,282` pixels
- **True Negatives (TN):** `117,012,191` pixels
- **Total Evaluated Pixels:** `117,964,800` pixels ($100 \times 768 \times 1536$)

---

## 7. Validation vs. Test Comparison

The table below contrasts the canonical validation set performance with the independent sealed test set benchmark:

| Metric | Canonical Validation Set (E56, $\tau=0.50$) | Sealed Test Set (100 Cases, $\tau=0.50$) | Absolute Difference ($\Delta = \text{Test} - \text{Val}$) |
| :--- | :---: | :---: | :---: |
| **Dice** | `65.623%` | `43.041%` | **`-22.582%`** |
| **IoU** | `49.854%` | `29.057%` | **`-20.797%`** |
| **Precision** | `69.009%` | `41.244%` | **`-27.765%`** |
| **Recall** | `63.649%` | `52.896%` | **`-10.753%`** |
| **Specificity** | `99.753%` | `99.630%` | **`-0.123%`** |

### Comparative Analysis:
1. **Generalization Gap:** The absolute difference in Macro Dice between the validation set (`65.62%`) and the sealed test set (`43.04%`) is `-22.58%`. 
2. **Error Decomposition:** The primary performance drop stems from increased false positives on out-of-distribution test anatomies (Macro Precision decreased from `69.01%` to `41.24%`, $\Delta = -27.77\%$), while boundary recall remained more resilient (`52.90%` vs `63.65%`, $\Delta = -10.75\%$).
3. **Background Discrimination:** Specificity remained exceptionally stable (`99.63%` vs `99.75%`, $\Delta = -0.12\%$), demonstrating that background false positive rates across non-canal tissue remain well controlled.

---

## 8. Per-Case Distribution Analysis

The per-case metrics across all 100 cases have been exported to [`outputs/diagnostics/EXP-MLUA-003_FINAL_TEST/EXP-MLUA-003_FINAL_TEST_PER_CASE.csv`](file:///c:/Users/devin/MLUA/outputs/diagnostics/EXP-MLUA-003_FINAL_TEST/EXP-MLUA-003_FINAL_TEST_PER_CASE.csv).

### Statistical Distribution Across 100 Test Cases:

| Metric | Mean | Median | Std Dev | Min (Worst) | Q25 | Q75 | Max (Best) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dice** | `43.041%` | `44.108%` | `17.872%` | `0.998%` | `28.773%` | `56.797%` | `77.936%` |
| **IoU** | `29.057%` | `28.298%` | `14.627%` | `0.502%` | `16.804%` | `39.665%` | `63.848%` |
| **Precision** | `41.244%` | `39.993%` | `21.494%` | `0.855%` | `24.548%` | `55.879%` | `91.385%` |
| **Recall** | `52.896%` | `55.148%` | `20.423%` | `1.200%` | `40.901%` | `66.940%` | `87.070%` |
| **Specificity** | `99.630%` | `99.719%` | `0.294%` | `98.432%` | `99.568%` | `99.805%` | `99.964%` |

### Top 5 Performing Test Cases:
| Case ID | Dice (%) | IoU (%) | Precision (%) | Recall (%) | Ground Truth Pixels |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1096** | `77.936%` | `63.848%` | `74.219%` | `82.044%` | 6,516 |
| **748** | `75.246%` | `60.316%` | `74.473%` | `76.036%` | 3,764 |
| **396** | `73.585%` | `58.209%` | `64.970%` | `84.834%` | 6,198 |
| **736** | `72.546%` | `56.924%` | `81.097%` | `65.626%` | 7,145 |
| **954** | `71.550%` | `55.700%` | `80.901%` | `64.136%` | 9,801 |

### Bottom 5 Performing Test Cases:
| Case ID | Dice (%) | IoU (%) | Precision (%) | Recall (%) | Ground Truth Pixels |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **392** | `0.998%` | `0.502%` | `0.855%` | `1.200%` | 3,668 |
| **1058** | `2.936%` | `1.489%` | `2.238%` | `4.268%` | 5,084 |
| **939** | `6.077%` | `3.134%` | `6.403%` | `5.782%` | 3,805 |
| **925** | `6.157%` | `3.176%` | `9.254%` | `4.613%` | 5,354 |
| **1091** | `14.135%` | `7.603%` | `10.461%` | `21.787%` | 1,634 |

---

## 9. Data Leakage & Test Protection Verification

We formally certify the strict separation and preservation of the test set:
1. **Training Separation:** The 100 test cases were never included in the training set (neither as labeled nor unlabeled samples).
2. **Checkpoint Selection Integrity:** The `EXP-MLUA-003_BEST.pth` checkpoint was selected exclusively on Epoch 56 validation Dice (`65.623%`) before accessing the test set.
3. **Threshold Integrity:** The threshold $\tau = 0.50$ was determined exclusively via validation sensitivity analysis before test evaluation.
4. **Single-Pass Evaluation:** The test evaluation script was executed as a single-pass inference run without iterative parameter adjustment or post-processing search.
5. **No Data Modification:** Zero test images or ground truth masks were altered in any way.

---

## 10. Limitations

1. **Research Benchmark Scope:** This evaluation constitutes an independent scientific research benchmark for semi-supervised mandibular canal segmentation on panoramic dental radiographs; it is not a clinical validation study and does not certify medical device performance.
2. **Domain Variability:** Significant performance variance across test cases (standard deviation of `17.87%` in Dice) highlights sensitivity to scanner variations, patient positioning, bone mineralization density, and radiographic artifacts.
3. **Fixed Resolution / Stride:** The 21-patch sliding window with fixed stride $192$ and overlap averaging was evaluated without test-time augmentation (TTA) or multi-scale ensembling to adhere strictly to the established protocol.

---

## 11. Final Scientific Conclusion

`EXP-MLUA-003` successfully completed training through 60 epochs with complete numerical stability and robust teacher-student EMA synchronization. On the independent sealed 100-case test set evaluated at the predetermined threshold $\tau = 0.50$, `EXP-MLUA-003` achieved a **Macro Test Dice of `43.041%`** (Micro Test Dice `43.391%`) with a **`0.0%` zero-prediction ratio** and **`99.630%` specificity**. All artifacts, per-case distributions, and summary tables have been fully generated and archived.

---

## 12. Final Decision Block

```text
============================================================
EXP-MLUA-003 FINAL TEST EVALUATION
============================================================

Checkpoint:
EXP-MLUA-003_BEST.pth

Epoch:
56

Threshold:
τ = 0.50

Test Cases:
100

Test Dice:
43.041%

Test IoU:
29.057%

Test Precision:
41.244%

Test Recall:
52.896%

Test Specificity:
99.630%

Validation Dice:
65.623%

Validation IoU:
49.854%

Validation Precision:
69.009%

Validation Recall:
63.649%

Validation Specificity:
99.753%

TEST SET EVALUATION:
[COMPLETE]

TRAINING:
[FROZEN]

THRESHOLD:
[FROZEN AT τ=0.50]

MODEL CHECKPOINT:
[FROZEN]

============================================================
```
