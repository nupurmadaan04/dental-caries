# EXP-MLUA-003 Controlled Micro-Extension Report (Epoch 76 → Epoch 78)

**Experiment ID**: `EXP-MLUA-003` (Micro-Extension Run)  
**Resumed Checkpoint**: `outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E75_LATEST.pth`  
**Execution Timestamp**: 2026-10-06 19:51:12  

---

## A. Resume Verification
- **Resume Checkpoint**: `outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E75_LATEST.pth`
- **Resume Epoch**: Epoch 75
- **Resume Global Step**: Step 9,900
- **Model State**: Student & Teacher state dictionaries loaded and verified 100% finite.
- **Optimizer State**: AdamW state loaded with complete momentum and variance buffers; 100% finite.
- **Scheduler State**: Polynomial learning rate scheduler state restored seamlessly (`last_epoch=75`).
- **EMA State**: Teacher EMA parameters and BatchNorm running statistics synchronized without reinitialization.

---

## B. E76–E78 Metrics Table

| Epoch | Global Step | Train Loss | Val Loss | Val Dice | Val IoU | Val Precision | Val Recall | Val Specificity | Zero-Pred Ratio | Learning Rate | Numerical Safety |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| E76 | 10032 | 0.4871 | 0.7369 | 69.245% | 54.180% | 69.438% | 69.594% | 99.718% | 2.0% | 6.50e-04 | 100% Finite (0 NaN/Inf) |
| E77 | 10164 | 0.4803 | 0.7357 | 69.234% | 54.341% | 78.401% | 62.727% | 99.840% | 2.0% | 6.46e-04 | 100% Finite (0 NaN/Inf) |
| E78 | 10296 | 0.4779 | 0.7287 | 71.768% | 57.060% | 78.918% | 67.061% | 99.834% | 0.0% | 6.41e-04 | 100% Finite (0 NaN/Inf) |

---

## C. Best Epoch
- **Extension Best Epoch (E76–E78)**: **Epoch 78** (Val Dice: **71.768%**, IoU: **57.060%**, Val Loss: **0.7287**)
- **Overall Trajectory Best Epoch (E1–E78)**: **Epoch 75** (Val Dice: **71.867%**)
- **New Validation Peak Achieved?**: **NO (Epoch 75 remains the validation peak)**

---

## D. Comparison with E75
Reference Epoch 75 Validation Metrics:
- **Dice**: 71.867% | **IoU**: 57.349% | **Precision**: 78.132% | **Recall**: 67.343% | **Specificity**: 99.824% | **Val Loss**: 0.7254

| Epoch | Val Dice (Delta vs E75) | Val IoU (Delta vs E75) | Val Precision (Delta vs E75) | Val Recall (Delta vs E75) | Val Specificity (Delta vs E75) | Val Loss (Delta vs E75) | Delta vs 71.12% | Exceeds 71.12%? | Exceeds E75? | New Best? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| E76 | 69.245% (-2.622 pp) | 54.180% (-3.169 pp) | 69.438% (-8.694 pp) | 69.594% (+2.251 pp) | 99.718% (-0.106 pp) | 0.7369 (+0.0115) | -1.875 pp | NO | NO | NO |
| E77 | 69.234% (-2.633 pp) | 54.341% (-3.008 pp) | 78.401% (+0.269 pp) | 62.727% (-4.616 pp) | 99.840% (+0.016 pp) | 0.7357 (+0.0103) | -1.886 pp | NO | NO | NO |
| E78 | 71.768% (-0.099 pp) | 57.060% (-0.289 pp) | 78.918% (+0.786 pp) | 67.061% (-0.282 pp) | 99.834% (+0.010 pp) | 0.7287 (+0.0033) | +0.648 pp | YES | NO | NO |

---

## E. Difference from 71.12% Published Reference
- **Published Research Reference**: 71.12% Dice
- **E75 Baseline Difference**: `+0.747` percentage points
- **E76 Difference**: `-1.875` percentage points
- **E77 Difference**: `-1.886` percentage points
- **E78 Difference**: `+0.648` percentage points

---

## F. Numerical Stability
- **NaN / Inf Incurrence**: **0 (Zero)** across all model weights, optimizer buffers, and loss terms.
- **Teacher/Student BatchNorm Synchronization**: Preserved and active across all batches (3 epochs $\times$ 132 batches/epoch = 396 batches).
- **Parameter Finiteness**: 100% verified after each epoch.

---

## G. Checkpoint Files Created
- `EXP-MLUA-003_E76.pth` & `EXP-MLUA-003_E76_LATEST.pth`
- `EXP-MLUA-003_E77.pth` & `EXP-MLUA-003_E77_LATEST.pth`
- `EXP-MLUA-003_E78.pth` & `EXP-MLUA-003_E78_LATEST.pth`
- No new `_BEST.pth` created (E75_BEST retained)

---

## H. E64 Integrity
- **Path**: `outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E64_BEST.pth`
- **Expected SHA256**: `167918b8375815668b5e784902e3e82a865b86053a610a312c775d52d473c526`
- **Observed Post-Run SHA256**: `167918b8375815668b5e784902e3e82a865b86053a610a312c775d52d473c526`
- **Integrity**: **100% Byte-Identical & Frozen**

---

## I. E75 Integrity
- **Path**: `outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E75_BEST.pth`
- **Expected SHA256**: `cb52ed6ec6911fab7f0081588fb63a6f0bad0e37b61c37f0ba9bd293a6bcedfb`
- **Observed Post-Run SHA256**: `cb52ed6ec6911fab7f0081588fb63a6f0bad0e37b61c37f0ba9bd293a6bcedfb`
- **Integrity**: **100% Byte-Identical & Frozen**

---

## J. Dataset Integrity
- **Labeled Training Set**: 530 patches (Untouched)
- **Unlabeled Training Set**: 1,859 patches (Untouched)
- **Validation Set**: 50 patches (Untouched, deterministic partition)
- **Total Training Samples**: 2,389 patches

---

## K. Sealed-Test Integrity
- **Sealed Test Set Access**: **STRICTLY ZERO ACCESS**
- **Test Images / Labels**: Not loaded or processed in any form.
- **E75 Sealed-Test Evaluation**: Preserved as the final untouched evaluation.

---

## L. Recommendation
**Chosen Directive**: **`STOP: E75 REMAINS BEST`**

**Analytical Rationale**:
None of the epochs in E76–E78 exceeded the Epoch 75 validation peak (71.867% Dice). Epoch 75 remains the optimal validated checkpoint.
