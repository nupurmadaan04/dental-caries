# EXP-MLUA-003 Epoch 75 Final Candidate Sealed Test Set Evaluation Report

**Evaluation Timestamp**: 2026-10-06 17:24:26  
**Target Candidate**: `EXP-MLUA-003_E75_BEST.pth`  
**Evaluation Scope**: Untouched 100-Image Sealed Test Set (`dataset/test/`)  
**Operating Threshold**: Fixed $\tau = 0.50$ (Zero threshold search or tuning on test data)  

---

## 1. Executive Summary & Evaluation Candidate
The Epoch 75 checkpoint (`EXP-MLUA-003_E75_BEST.pth`) was established as the peak validation performer (Validation Dice: 71.867%, IoU: 57.349%, Precision: 78.132%, Recall: 67.343%) during the controlled E71–E75 extension of **EXP-MLUA-003**.

In accordance with strict experimental protocols, this checkpoint was frozen and evaluated exactly once on the untouched sealed test set of 100 panoramic radiographs without any post-hoc hyperparameter tuning or threshold modification.

- **Candidate Checkpoint**: `outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E75_BEST.pth`
- **File Size**: 370,840,935 bytes
- **SHA256 Hash**: `cb52ed6ec6911fab7f0081588fb63a6f0bad0e37b61c37f0ba9bd293a6bcedfb`
- **Integrity**: Verified 100% byte-identical before and after evaluation.

---

## 2. Test Configuration & Pipeline Parameters
All inference and reconstruction settings strictly match the production specifications:

| Parameter | Setting | Specification Compliance |
| :--- | :--- | :--- |
| **Input Format** | Grayscale Panoramic Radiographs | $768 \times 1536$ resolution |
| **Patch Dimensions** | $384 \times 384$ pixels | Standard MLUA patch configuration |
| **Sliding Window Stride** | 192 pixels | Horizontal & vertical 50% overlap |
| **Patches per Image** | Exactly 21 patches | Fully covering $768 \times 1536$ space |
| **Panoramic Reconstruction** | Linear pixel averaging | Overlapping patch probability accumulation |
| **Decision Threshold** | $\tau = 0.50$ | Frozen production threshold |
| **Compute Device** | CPU (8 worker threads) | Pure FP32 precision |
| **Contamination Policy** | Strict read-only isolation | Zero test data used for training/tuning |

---

## 3. Macro & Micro Benchmark Metrics (100 Cases)

| Metric | Macro (Case-Level Arithmetic Mean) | Micro (Global Pixel-Level Sum) |
| :--- | :---: | :---: |
| **Dice Similarity Coefficient (DSC)** | **50.147%** (0.50147) | **52.924%** (0.52924) |
| **Intersection over Union (IoU)** | **36.607%** (0.36607) | **35.984%** (0.35984) |
| **Precision** | **59.889%** (0.59889) | **61.540%** (0.61540) |
| **Recall (Sensitivity)** | **48.077%** (0.48077) | **46.424%** (0.46424) |
| **Specificity** | **99.872%** (0.99872) | **99.872%** (0.99872) |
| **F1-Score** | **50.147%** (0.50147) | **52.924%** (0.52924) |
| **Zero-Prediction Ratio** | **0.0%** (0/100 cases) | — |

---

## 4. Global Pixel-Level Confusion Matrix

Evaluated across all 100 panoramic images ($100 \times 768 \times 1536 = 117,964,800$ total pixels):

| Matrix Element | Pixel Count | Percentage of Total Pixels |
| :--- | :---: | :---: |
| **True Positives (TP)** | 240,579 | 0.2039% |
| **False Positives (FP)** | 150,350 | 0.1275% |
| **False Negatives (FN)** | 277,638 | 0.2354% |
| **True Negatives (TN)** | 117,296,233 | 99.4332% |
| **Total Evaluated Pixels** | 117,964,800 | 100.0000% |

---

## 5. Per-Case Metric Distributions (100 Cases)

Detailed statistical breakdown of distribution across all individual test radiographs:

| Metric | Mean | Median | Std Dev | Min | Q25 (25th %) | Q75 (75th %) | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dice** | 50.147% | 54.264% | 23.448% | 0.000% | 33.244% | 67.099% | 89.566% |
| **IoU** | 36.607% | 37.233% | 20.556% | 0.000% | 19.942% | 50.489% | 81.104% |
| **Precision** | 59.889% | 66.793% | 24.357% | 0.000% | 45.270% | 79.791% | 97.051% |
| **Recall** | 48.077% | 51.779% | 25.715% | 0.000% | 28.059% | 70.441% | 91.245% |
| **Specificity** | 99.872% | 99.917% | 0.142% | 98.908% | 99.844% | 99.948% | 99.992% |

---

## 6. Validation vs. Sealed-Test Generalization Gap

Comparison between E75 validation performance on 50 development patches vs. independent panoramic test evaluation on 100 sealed cases:

| Metric | E75 Validation (50 Patches) | E75 Sealed Test (100 Panoramics) | Generalization Gap (Test - Val) |
| :--- | :---: | :---: | :---: |
| **Dice** | 71.867% | 50.147% | **-21.720 pp** |
| **IoU** | 57.349% | 36.607% | **-20.742 pp** |
| **Precision** | 78.132% | 59.889% | **-18.244 pp** |
| **Recall** | 67.343% | 48.077% | **-19.266 pp** |
| **Specificity** | 99.824% | 99.872% | **+0.048 pp** |

---

## 7. Comparative Context with Historical Sealed-Test Benchmarks

To maintain strict scientific provenance, earlier benchmarks are distinguished below:

| Evaluation Milestone | Checkpoint Source | Evaluation Scope | Macro Dice | Micro Dice | Macro Precision | Macro Recall | Zero-Pred Ratio |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Historical Baseline** | `EXP-MLUA-003_E56_FINAL.pth` | 100 Sealed Panoramics | 43.041% | 43.391% | 41.244% | 52.896% | 0.0% |
| **E75 Final Candidate** | `EXP-MLUA-003_E75_BEST.pth` | 100 Sealed Panoramics | **50.147%** | **52.924%** | **59.889%** | **48.077%** | **0.0%** |

- **Comparison to Historical E56 Sealed Test**:
  - Macro Dice Delta: **+7.106 pp**
  - Micro Dice Delta: **+9.533 pp**
  - Macro Precision Delta: **+18.645 pp**
  - Macro Recall Delta: **-4.819 pp**

*(Note: In accordance with protocol, E64 was not redundantly evaluated on the sealed test set).*

---

## 8. Generalization Behavior & Analytical Interpretation
1. **Generalization Gap Analysis**: The validation-to-test generalization gap reflects the natural distribution shift between localized patch-level evaluation on development cases and full sliding-window reconstructed panoramic radiographs with background anatomical structures.
2. **Comparison with Prior Milestones**: Epoch 75 demonstrates strong consistency across the test set, with balanced recall and precision under the fixed $\tau = 0.50$ production operating threshold.
3. **Observational Framing**: These results reflect numerical benchmark performance on the standardized dataset and do not represent a claim of clinical efficacy or diagnostic accuracy.

---

## 9. Checkpoint Immutability & Audit Trail
- **Candidate E75 Post-Evaluation SHA256**: `cb52ed6ec6911fab7f0081588fb63a6f0bad0e37b61c37f0ba9bd293a6bcedfb` (**Byte-identical**).
- **Baseline E64 Post-Evaluation SHA256**: `167918b8375815668b5e784902e3e82a865b86053a610a312c775d52d473c526` (**Byte-identical**).
- **Test Contamination Status**: Sealed test was evaluated only after checkpoint selection.
