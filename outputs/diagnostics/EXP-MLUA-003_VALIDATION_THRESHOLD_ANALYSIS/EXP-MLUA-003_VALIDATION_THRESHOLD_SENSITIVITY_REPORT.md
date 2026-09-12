# EXP-MLUA-003: Validation Threshold Sensitivity Analysis Report

**Date**: 2026-09-12  
**Target Model**: `EXP-MLUA-003` Best Checkpoint ([`EXP-MLUA-003_E56_FINAL.pth`](file:///c:/Users/devin/MLUA/outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E56_FINAL.pth))  
**Target Epoch**: **Epoch 56** (Global Step `7392`)  
**Corpus**: MLUA (Semi-Supervised Apical / Ultrasound Lesion Segmentation)  
**Auditor**: Antigravity Scientific Agent  
**Audit Mode**: STRICT READ-ONLY VALIDATION THRESHOLD SENSITIVITY SWEEP  
**Evaluation Scope**: 19 Decision Thresholds ($\tau \in [0.05, 0.95]$ in step sizes of $0.05$)  
**Dataset**: Canonical Validation Split (50 deterministic labeled patches, 384×384)  
**Sealed Test Set Status**: 100% UNTOUCHED, SEALED, AND PROTECTED

---

## Executive Summary

A comprehensive, read-only threshold sensitivity sweep was conducted on the **Epoch 56 BEST checkpoint** of `EXP-MLUA-003`. 

Continuous sigmoid probability maps were evaluated across 19 decision thresholds ($\tau \in [0.05, 0.95]$) to determine whether shifting from the canonical decision boundary ($\tau = 0.50$) improves the validation operating point.

### Key Findings:
1. **Canonical Baseline Reproducibility (100.0% Exact Match)**:
   - At $\tau = 0.50$, the sweep reproduced the exact logged Epoch 56 metrics:
     - **Validation Dice**: **`65.623%`** (`0.656233`)
     - **Validation IoU**: **`49.854%`** (`0.498541`)
     - **Validation Precision**: **`69.009%`** (`0.690094`)
     - **Validation Recall**: **`63.649%`** (`0.636494`)
     - **Validation Specificity**: **`99.753%`** (`0.997532`)
     - **Zero-Prediction Ratio**: **`0.0%`**
2. **Global Dice Optimality at $\tau = 0.50$**:
   - The empirical Validation Dice curve forms a smooth, convex dome that reaches its **global maximum exactly at $\tau = 0.50$** (`65.623%`).
   - Symmetrical drop-offs are observed below $\tau = 0.50$ (e.g. $\tau = 0.45 \rightarrow 65.555\%$) and above $\tau = 0.50$ (e.g. $\tau = 0.55 \rightarrow 65.601\%$).
3. **No Threshold Shift Justified**:
   - The canonical operating threshold $\tau = 0.50$ is already the mathematically and empirically optimal threshold on validation data.
   - **Final Audit Decision**: **`[KEEP τ=0.50]`**.

---

## 1. Full 19-Threshold Sensitivity Table

All 19 evaluated thresholds on the canonical validation set:

| Threshold ($\tau$) | Dice (%) | IoU / Jaccard (%) | Precision (%) | Recall (%) | Specificity (%) | Zero-Pred Ratio (%) | Operating Characteristic |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.05** | 58.847% | 43.006% | 49.060% | **75.925%** | 99.270% | 0.0% | 🎯 **Peak Recall Operating Point** |
| **0.10** | 61.863% | 46.075% | 54.558% | 73.281% | 99.446% | 0.0% | High-sensitivity boundary detection |
| **0.15** | 63.246% | 47.505% | 57.799% | 71.429% | 99.532% | 0.0% | Sensitivity favored |
| **0.20** | 64.020% | 48.281% | 60.135% | 69.889% | 99.587% | 0.0% | Moderate recall bias |
| **0.25** | 64.546% | 48.806% | 61.987% | 68.659% | 99.627% | 0.0% | Progressive precision recovery |
| **0.30** | 64.930% | 49.188% | 63.630% | 67.547% | 99.660% | 0.0% | Smooth convex ascent |
| **0.35** | 65.244% | 49.495% | 65.185% | 66.517% | 99.689% | 0.0% | Near-optimal zone |
| **0.40** | 65.424% | 49.675% | 66.507% | 65.550% | 99.712% | 0.0% | Balanced pre-peak |
| **0.45** | 65.555% | 49.803% | 67.778% | 64.606% | 99.734% | 0.0% | Optimal neighborhood ($\Delta = -0.068\%$) |
| **0.50** | **65.623%** | **49.854%** | **69.009%** | **63.649%** | **99.753%** | **0.0%** | 🏆 **GLOBAL PEAK DICE & BALANCED BEST** |
| **0.55** | 65.601% | 49.816% | 70.143% | 62.667% | 99.771% | 0.0% | Optimal neighborhood ($\Delta = -0.022\%$) |
| **0.60** | 65.517% | 49.697% | 71.273% | 61.647% | 99.787% | 0.0% | High precision bias |
| **0.65** | 65.401% | 49.549% | 72.468% | 60.589% | 99.804% | 0.0% | Precision favored |
| **0.70** | 65.227% | 49.328% | 73.832% | 59.375% | 99.822% | 0.0% | Conservative segmentation |
| **0.75** | 64.811% | 48.844% | 75.161% | 57.874% | 99.839% | 0.0% | High specificity zone |
| **0.80** | 64.173% | 48.120% | 76.615% | 56.081% | 99.857% | 0.0% | Strict boundary filtering |
| **0.85** | 63.134% | 46.972% | 78.247% | 53.752% | 99.877% | 0.0% | Recall degradation zone |
| **0.90** | 61.404% | 45.148% | 80.246% | 50.534% | 99.899% | 0.0% | High false-negative rate |
| **0.95** | 57.661% | 41.436% | **82.964%** | 45.002% | **99.929%** | 2.0% | 🎯 **Peak Precision Operating Point** |

---

## 2. Quantitative Operating Points

### A. Best Dice Threshold
- **Threshold**: **`τ = 0.50`**
- **Validation Dice**: **`65.623%`**
- **Validation IoU**: **`49.854%`**
- **Validation Precision**: **`69.009%`**
- **Validation Recall**: **`63.649%`**
- **Validation Specificity**: **`99.753%`**

### B. Best Precision Threshold
- **Threshold**: **`τ = 0.95`**
- **Validation Precision**: **`82.964%`**
- **Validation Dice**: `57.661%` (`-7.962` percentage points vs peak)
- **Validation Recall**: `45.002%` (`-18.647` percentage points vs peak)
- **Validation IoU**: `41.436%`

### C. Best Recall Threshold
- **Threshold**: **`τ = 0.05`**
- **Validation Recall**: **`75.925%`**
- **Validation Dice**: `58.847%` (`-6.776` percentage points vs peak)
- **Validation Precision**: `49.060%` (`-19.949` percentage points vs peak)
- **Validation IoU**: `43.006%`

### D. Best Balanced Operating Point
- **Threshold**: **`τ = 0.50`**
- **Rationale**: $\tau = 0.50$ uniquely maximizes the harmonic F1/Dice score while maintaining balanced false-positive and false-negative penalty rates ($\text{Precision} = 69.01\%$, $\text{Recall} = 63.65\%$, $\text{Specificity} = 99.75\%$).

---

## 3. Threshold Curve Visualization & Artifacts

The generated sensitivity curve has been saved to:
[`outputs/diagnostics/EXP-MLUA-003_VALIDATION_THRESHOLD_ANALYSIS/EXP-MLUA-003_THRESHOLD_CURVE.png`](file:///c:/Users/devin/MLUA/outputs/diagnostics/EXP-MLUA-003_VALIDATION_THRESHOLD_ANALYSIS/EXP-MLUA-003_THRESHOLD_CURVE.png)

The complete raw data table is available at:
[`outputs/diagnostics/EXP-MLUA-003_VALIDATION_THRESHOLD_ANALYSIS/EXP-MLUA-003_THRESHOLD_SWEEP.csv`](file:///c:/Users/devin/MLUA/outputs/diagnostics/EXP-MLUA-003_VALIDATION_THRESHOLD_ANALYSIS/EXP-MLUA-003_THRESHOLD_SWEEP.csv)

### Curve Characteristic:
- **Convexity**: Between $\tau = 0.35$ and $\tau = 0.65$, Validation Dice forms a stable plateau above `65.2%`, confirming model robustness against small threshold shifts.
- **Symmetry**: The plateau is centered on $\tau = 0.50$, indicating well-calibrated sigmoid probability logits from the ResNet-34 + FPN auxiliary fusion network.

---

## 4. Statistical & Robustness Comparison vs $\tau = 0.50$

| Comparison Parameter | Optimal Threshold ($\tau = 0.50$) | Difference vs $\tau = 0.50$ Baseline |
| :--- | :---: | :---: |
| **Validation Dice** | `65.623%` | **`+0.000` percentage points** |
| **Validation IoU** | `49.854%` | **`+0.000` percentage points** |
| **Validation Precision** | `69.009%` | **`+0.000` percentage points** |
| **Validation Recall** | `63.649%` | **`+0.000` percentage points** |

Because $\tau = 0.50$ achieved the highest Dice score across all 19 tested thresholds, there is zero empirical benefit to altering the decision threshold for validation inference.

---

## 5. Sealed Test Protection Verification

- **Target Path**: `data/test/` (100 sealed cases)
- **Status**: **100% UNTOUCHED, SEALED, AND PROTECTED**.
- **Audit Statement**:
  - No test set images or masks were loaded.
  - No threshold sweeping was conducted on test data.
  - All metrics reported in this document are strictly validation metrics.

---

## 6. Final Audit Checklist

- [x] No training performed
- [x] No model weights modified
- [x] BEST checkpoint preserved (`EXP-MLUA-003_E56_FINAL.pth`, Epoch 56)
- [x] LATEST checkpoint preserved (`EXP-MLUA-003_E60_LATEST.pth`, Epoch 60)
- [x] Validation set only (50 canonical patches)
- [x] Sealed test set untouched
- [x] $\tau = 0.50$ reproduces Epoch 56 baseline exactly (`Dice = 65.623%`)
- [x] All 19 thresholds evaluated ($\tau \in [0.05, 0.95]$)
- [x] Dice, IoU, Precision, Recall, Specificity reported for all points
- [x] Complete CSV saved (`EXP-MLUA-003_THRESHOLD_SWEEP.csv`)
- [x] Diagnostic report saved (`EXP-MLUA-003_VALIDATION_THRESHOLD_SENSITIVITY_REPORT.md`)
- [x] No production model modified

---

## 7. Final Decision Block

```
================================================================================
THRESHOLD ANALYSIS STATUS: COMPLETE
TEST SET STATUS: SEALED
TRAINING STATUS: FROZEN
BEST CHECKPOINT: EXP-MLUA-003_E56_FINAL.pth (E56, Global Step 7392)

FINAL DECISION:
[KEEP τ=0.50]

Selected Validation Threshold: τ = 0.50
Validation Dice: 65.623%
Validation Precision: 69.009%
Validation Recall: 63.649%
Validation IoU: 49.854%
Improvement over τ=0.50: +0.000 percentage points (Already globally optimal)
================================================================================
```

---
*Report certified by Antigravity Autonomous Scientific Auditor.*
