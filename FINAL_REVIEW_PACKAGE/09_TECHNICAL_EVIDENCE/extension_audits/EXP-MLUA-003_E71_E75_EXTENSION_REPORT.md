# EXP-MLUA-003 Training Continuation Report (Epoch 71 → Epoch 75)
**Experiment ID**: `EXP-MLUA-003` (Extension Run)  
**Resumed Checkpoint**: `outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E70_LATEST.pth`  
**Execution Timestamp**: 2026-10-06 13:33:25  

---

## 1. Executive Purpose & Scope
This continuation strictly extends the existing **EXP-MLUA-003** training trajectory from completed **Epoch 70** to **Epoch 75** (5 additional epochs).
- **Experiment Identity**: Preserved as `EXP-MLUA-003` (no new experiment ID created).
- **Restart Prevention**: Initialized directly from the serialized optimizer, scheduler, model student, and model teacher state at Epoch 70 (`global_step=9240`).
- **Sealed Test Set**: **Untouched** (100-case sealed benchmark strictly isolated; evaluation is validation-only at $\tau = 0.50$).

---

## 2. Configuration & Controlled Parameter Verification
All model, data, optimizer, and semi-supervised hyperparameters were maintained 100% identical to the EXP-MLUA-003 specification:

| Component | Setting | Status |
| :--- | :--- | :--- |
| **Architecture** | ResNet-34 Encoder + FPN Decoder + 4 Aux Heads | Identical |
| **Dataset** | DC1000 (530 Labeled / 1,859 Unlabeled patches) | Identical |
| **Patch Resolution / Normalization** | $384 \times 384$, Grayscale $[0, 1]$ | Identical |
| **Random Seed** | 42 | Identical |
| **Optimizer & Schedule** | AdamW (lr=0.001, wd=0.01), Poly LR ($p=0.9, \text{max}=200$) | Identical |
| **Semi-Supervised Mechanism** | MLUA Dual-Teacher MC-Dropout ($T=8$, $\sigma=0.01$) | Identical |
| **EMA Synchronization** | Parameter EMA + BatchNorm Buffer EMA ($\theta=0.99$) | Strictly Preserved |
| **Validation Threshold** | $\tau = 0.50$ | Identical |

---

## 3. Epoch 71–75 Validation Metrics Table

| Epoch | Train Loss | Val Loss | Val Dice | Val IoU | Val Precision | Val Recall | Val Specificity | Zero-Pred Ratio | Max FG Prob | Learning Rate | Global Step |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| E71 | 0.5021 | 0.7466 | 67.758% | 52.659% | 70.474% | 66.028% | 99.748% | 0.0% | 1.0000 | 6.74e-04 | 9372 |
| E72 | 0.4927 | 0.7330 | 70.602% | 55.653% | 85.017% | 61.229% | 99.897% | 0.0% | 1.0000 | 6.69e-04 | 9504 |
| E73 | 0.4922 | 0.7344 | 69.877% | 54.964% | 71.894% | 69.054% | 99.764% | 0.0% | 1.0000 | 6.65e-04 | 9636 |
| E74 | 0.4938 | 0.7484 | 69.180% | 53.996% | 86.583% | 59.048% | 99.912% | 4.0% | 1.0000 | 6.60e-04 | 9768 |
| E75 | 0.4950 | 0.7254 | 71.867% | 57.349% | 78.132% | 67.343% | 99.824% | 2.0% | 1.0000 | 6.55e-04 | 9900 |

---

## 4. Performance Trajectory & Historical Comparison

| Checkpoint / Milestone | Epoch | Global Step | Val Dice | Val IoU | Val Precision | Val Recall | Val Specificity | Val Loss | Zero-Pred % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical Baseline (E56)** | 56 | 7,392 | 65.623% | 49.854% | 69.009% | 63.649% | 99.753% | 0.7639 | 0.0% |
| **Reference BEST (E64)** | 64 | 8,448 | 69.386% | 54.326% | 74.689% | 66.415% | 99.784% | 0.7471 | 0.0% |
| **Resume Checkpoint (E70)** | 70 | 9,240 | 65.438% | 49.812% | 64.893% | 67.590% | 99.647% | 0.7701 | 0.0% |
| **E71–E75 Best Epoch** | 75 | 9900 | 71.867% | 57.349% | 78.132% | 67.343% | 99.824% | 0.7254 | 2.0% |
| **Final Checkpoint (E75)** | 75 | 9900 | 71.867% | 57.349% | 78.132% | 67.343% | 99.824% | 0.7254 | 2.0% |

### Absolute Differences Against Canonical E64 BEST (Percentage Points / Direct Delta)
- **Val Dice Difference (E75 - E64)**: `+2.481` percentage points
- **Val IoU Difference (E75 - E64)**: `+3.023` percentage points
- **Val Precision Difference (E75 - E64)**: `+3.443` percentage points
- **Val Recall Difference (E75 - E64)**: `+0.928` percentage points
- **Val Specificity Difference (E75 - E64)**: `+0.040` percentage points
- **Val Loss Difference (E75 - E64)**: `-0.0217`
- **Zero-Prediction Ratio Difference (E75 - E64)**: `+2.00` percentage points
- **Extension Best vs E64 Dice Delta**: `+2.481` percentage points

### Absolute Differences Against Resume Checkpoint E70
- **Val Dice Difference (E75 - E70)**: `+6.429` percentage points
- **Val IoU Difference (E75 - E70)**: `+7.537` percentage points
- **Val Precision Difference (E75 - E70)**: `+13.239` percentage points
- **Val Recall Difference (E75 - E70)**: `-0.247` percentage points
- **Val Loss Difference (E75 - E70)**: `-0.0447`

---

## 5. Observable Trajectory Analysis
1. **Did Dice improve?**: YES, increased by +2.481 pp compared to E64 (compared to E70: `+6.429` pp).
2. **Did IoU improve?**: YES (`+3.023` pp vs E64).
3. **Did precision improve?**: YES (`+3.443` pp vs E64).
4. **Did recall improve?**: YES (`+0.928` pp vs E64).
5. **Did validation loss improve?**: YES (`-0.0217` vs E64).
6. **Did specificity change?**: `+0.040` pp difference vs E64.
7. **Did zero-prediction ratio change?**: `+2.00` pp difference vs E64 (E75 is `2.0%`).
8. **Did performance stabilize or degrade?**: Performance improved from E70.

---

## 6. Numerical Stability Audit
- **NaN / Inf Incurrence**: **0 (Zero)** across all epochs, steps, parameters, and buffers.
- **Teacher/Student BatchNorm Buffers**: Fully synchronized across all 660 extension batches (5 epochs $\times$ 132 batches/epoch).
- **Loss and Gradient Bounds**: 100% finite and bounded throughout Steps 9,241 to 9900.

---

## 7. Checkpoint Selection & Canon Preservation
- **Extension Best Epoch (E71–E75)**: **Epoch 75** (Dice: `71.867%`)
- **Canonical Reference Best (E64)**: **Epoch 64** (Dice: `69.386%`)
- **Did Any Epoch in E71–E75 Surpass E64?**: **YES**
- **Canonical Model Decision**: Extension candidate Epoch 75 achieved a new validation best.
