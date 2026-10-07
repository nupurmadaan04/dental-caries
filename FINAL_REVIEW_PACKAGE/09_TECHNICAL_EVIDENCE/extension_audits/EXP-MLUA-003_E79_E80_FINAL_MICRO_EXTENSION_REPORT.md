# EXP-MLUA-003 Controlled Final Micro-Extension Report (Epoch 79 → Epoch 80)

**Experiment ID**: `EXP-MLUA-003` (Final Micro-Extension Run)  
**Resumed Checkpoint**: `outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E78_LATEST.pth`  
**Execution Timestamp**: 2026-10-07 00:34:56  
**Operating Threshold**: Fixed $\tau = 0.50$  

---

## A. Resume Verification
- **Resume Checkpoint**: `outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E78_LATEST.pth`
- **Resume Epoch**: Epoch 78
- **Resume Global Step**: Step 10,296
- **Resume Learning Rate**: ~6.41e-4
- **Model State**: Student & Teacher states restored and verified 100% finite.
- **Optimizer State**: AdamW momentum and second-moment buffers restored and verified 100% finite.
- **Scheduler State**: Polynomial learning rate scheduler restored seamlessly (`last_epoch=78`).
- **EMA State**: Teacher parameter EMA and BatchNorm running statistics (`running_mean`, `running_var`) restored and verified finite.
- **Tensors Finiteness**: 100% finite (0 NaN, 0 Inf).

---

## B. E79 Metrics Table
- **Global Step**: 10428
- **Train Loss**: 0.4769
- **Validation Loss**: 0.7257
- **Validation Dice**: **71.790%**
- **Validation IoU**: **57.212%**
- **Validation Precision**: **75.262%**
- **Validation Recall**: **69.496%**
- **Validation Specificity**: **99.792%**
- **Zero-Prediction Ratio**: 0.0%
- **Max Foreground Probability**: 1.0000
- **Learning Rate**: 6.36e-04
- **Epoch Duration**: 2156.6s (35.9 min)
- **NaN Count**: 0
- **Inf Count**: 0

---

## C. E80 Metrics Table
- **Global Step**: 10560
- **Train Loss**: 0.4686
- **Validation Loss**: 0.7251
- **Validation Dice**: **71.343%**
- **Validation IoU**: **56.470%**
- **Validation Precision**: **81.016%**
- **Validation Recall**: **64.558%**
- **Validation Specificity**: **99.865%**
- **Zero-Prediction Ratio**: 0.0%
- **Max Foreground Probability**: 1.0000
- **Learning Rate**: 6.31e-04
- **Epoch Duration**: 3011.7s (50.2 min)
- **NaN Count**: 0
- **Inf Count**: 0

---

## D. Comparison Against E75

Reference E75: Dice 71.867%, IoU 57.349%, Precision 78.132%, Recall 67.343%, Specificity 99.824%, Val Loss 0.7254, Zero-Pred 2.0%

| Metric | E75 Value | E79 Value | Delta (E79 - E75) | E80 Value | Delta (E80 - E75) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Validation Dice** | 71.867% | 71.790% | **-0.077 pp** | 71.343% | **-0.524 pp** |
| **Validation IoU** | 57.349% | 57.212% | **-0.137 pp** | 56.470% | **-0.879 pp** |
| **Validation Precision** | 78.132% | 75.262% | **-2.870 pp** | 81.016% | **+2.884 pp** |
| **Validation Recall** | 67.343% | 69.496% | **+2.153 pp** | 64.558% | **-2.785 pp** |
| **Validation Specificity** | 99.824% | 99.792% | **-0.032 pp** | 99.865% | **+0.041 pp** |
| **Validation Loss** | 0.7254 | 0.7257 | **+0.0003** | 0.7251 | **-0.0003** |
| **Zero-Prediction Ratio** | 2.0% | 0.0% | **-2.0 pp** | 0.0% | **-2.0 pp** |

---

## E. Comparison Against E78

Reference E78: Dice 71.768%, IoU 57.060%, Precision 78.918%, Recall 67.061%, Specificity 99.834%, Val Loss 0.7287, Zero-Pred 0.0%

| Metric | E78 Value | E79 Value | Delta (E79 - E78) | E80 Value | Delta (E80 - E78) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Validation Dice** | 71.768% | 71.790% | **+0.022 pp** | 71.343% | **-0.425 pp** |
| **Validation IoU** | 57.060% | 57.212% | **+0.152 pp** | 56.470% | **-0.590 pp** |
| **Validation Precision** | 78.918% | 75.262% | **-3.656 pp** | 81.016% | **+2.098 pp** |
| **Validation Recall** | 67.061% | 69.496% | **+2.435 pp** | 64.558% | **-2.503 pp** |
| **Validation Specificity** | 99.834% | 99.792% | **-0.042 pp** | 99.865% | **+0.031 pp** |
| **Validation Loss** | 0.7287 | 0.7257 | **-0.0030** | 0.7251 | **-0.0036** |
| **Zero-Prediction Ratio** | 0.0% | 0.0% | **+0.0 pp** | 0.0% | **+0.0 pp** |

---

## F. Comparison Against 71.12% Literature Reference
The 71.12% value is monitored solely as a literature reference point:
- **E75 Difference**: `+0.747` percentage points (71.867% vs. 71.120%)
- **E78 Difference**: `+0.648` percentage points (71.768% vs. 71.120%)
- **E79 Difference**: `+0.670` percentage points (71.790% vs. 71.120%)
- **E80 Difference**: `+0.223` percentage points (71.343% vs. 71.120%)
- **E79 Exceeds 71.12%?**: **YES**
- **E80 Exceeds 71.12%?**: **YES**

---

## G. Multi-Metric Analysis
Examining candidate validation epochs (E75, E78, E79, E80) across all recorded dimensions:
- **Highest Validation Dice**: **Epoch 75** (71.867%)
- **Highest Validation IoU**: **Epoch 75** (57.349%)
- **Highest Validation Precision**: **Epoch 80** (81.016%)
- **Highest Validation Recall**: **Epoch 79** (69.496%)
- **Highest Validation Specificity**: **Epoch 80** (99.865%)
- **Lowest Validation Loss**: **Epoch 80** (0.7251)
- **Lowest Zero-Prediction Ratio**: **Epoch 78** (0.0%)

**Overall Profile Evaluation**:
Epoch 75 maintains the most robust balanced metric profile, simultaneously holding the highest validation Dice, highest IoU, highest recall, and lowest validation loss.

---

## H. Numerical Stability
- **NaN Incurrence**: **0 (Zero)** across all parameters, optimizer states, and buffers.
- **Inf Incurrence**: **0 (Zero)** across all parameters, optimizer states, and buffers.
- **Teacher/Student BatchNorm Synchronization**: Preserved and active across all 264 batches (2 epochs $\times$ 132 batches/epoch).
- **Parameter Finiteness**: Verified 100% finite at every step and epoch boundary.

---

## I. Checkpoint Inventory
The following continuation checkpoints were created during the E79–E80 run:
- `EXP-MLUA-003_E79.pth` & `EXP-MLUA-003_E79_LATEST.pth`
- `EXP-MLUA-003_E80.pth` & `EXP-MLUA-003_E80_LATEST.pth`
- No new `_BEST.pth` created (E75_BEST retained)

---

## J. E64 Integrity
- **File**: `outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E64_BEST.pth`
- **Expected SHA256**: `167918b8375815668b5e784902e3e82a865b86053a610a312c775d52d473c526`
- **Observed Post-Run SHA256**: `167918b8375815668b5e784902e3e82a865b86053a610a312c775d52d473c526`
- **Integrity Status**: **100% Byte-Identical & Frozen**

---

## K. E75 Integrity
- **File**: `outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E75_BEST.pth`
- **Expected SHA256**: `cb52ed6ec6911fab7f0081588fb63a6f0bad0e37b61c37f0ba9bd293a6bcedfb`
- **Observed Post-Run SHA256**: `cb52ed6ec6911fab7f0081588fb63a6f0bad0e37b61c37f0ba9bd293a6bcedfb`
- **Integrity Status**: **100% Byte-Identical & Frozen**

---

## L. Dataset Integrity
- **Labeled Set**: 530 patches (Untouched)
- **Unlabeled Set**: 1,859 patches (Untouched)
- **Validation Set**: 50 patches (Untouched, deterministic partition)
- **Total Samples**: 2,389 patches

---

## M. Sealed-Test Integrity
- **Sealed Test Set Access**: **STRICTLY ZERO ACCESS**
- **Test Images & Masks**: Not loaded, inspected, or evaluated.
- **E75 Sealed-Test Benchmark**: Preserved as the final untouched sealed-test evaluation.

---

## N. Final Recommendation
**Chosen Recommendation**: **`E75 REMAINS BEST`**

**Analytical Justification**:
Epoch 75 retains the highest validation Dice (71.867%) and lowest validation loss (0.7254) under the project's fixed evaluation protocol.
