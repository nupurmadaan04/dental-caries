# EXP-MLUA-003 Training Continuation Report (Epoch 61 → Epoch 70)
**Experiment ID**: `EXP-MLUA-003` (Extension Run)  
**Resumed Checkpoint**: `outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E60_LATEST.pth`  
**Execution Timestamp**: 2026-10-03 03:10:31  

---

## 1. Executive Purpose & Scope
This continuation strictly extends the existing **EXP-MLUA-003** training trajectory from completed **Epoch 60** to **Epoch 70** (10 additional epochs). 
- **Experiment Identity**: Preserved as `EXP-MLUA-003` (no new experiment ID created).
- **Restart Prevention**: Initialized directly from the serialized optimizer, scheduler, model student, and model teacher state at Epoch 60 (`global_step=7920`).
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

## 3. Epoch 61–70 Validation Metrics Table

| Epoch | Train Loss | Val Loss | Val Dice | Val IoU | Val Precision | Val Recall | Val Specificity | Zero-Pred Ratio | Max FG Prob | Learning Rate | Global Step |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| E61 | 0.5200 | 0.7833 | 59.819% | 45.433% | 72.518% | 51.959% | 99.859% | 0.0% | 1.0000 | 7.21e-04 | 8052 |
| E62 | 0.5111 | 0.7573 | 68.047% | 52.770% | 78.330% | 61.368% | 99.834% | 4.0% | 1.0000 | 7.16e-04 | 8184 |
| E63 | 0.5076 | 0.7711 | 64.924% | 49.898% | 76.377% | 57.838% | 99.859% | 0.0% | 1.0000 | 7.11e-04 | 8316 |
| E64 | 0.5121 | 0.7471 | 69.386% | 54.326% | 74.689% | 66.415% | 99.784% | 0.0% | 1.0000 | 7.07e-04 | 8448 |
| E65 | 0.5142 | 0.7642 | 65.059% | 49.653% | 79.586% | 56.796% | 99.871% | 0.0% | 1.0000 | 7.02e-04 | 8580 |
| E66 | 0.5109 | 0.7623 | 64.602% | 48.963% | 76.381% | 57.060% | 99.839% | 2.0% | 1.0000 | 6.97e-04 | 8712 |
| E67 | 0.5052 | 0.7407 | 67.277% | 51.600% | 80.810% | 58.475% | 99.877% | 2.0% | 1.0000 | 6.93e-04 | 8844 |
| E68 | 0.5060 | 0.7632 | 65.043% | 50.591% | 76.380% | 58.187% | 99.843% | 0.0% | 1.0000 | 6.88e-04 | 8976 |
| E69 | 0.5075 | 0.7522 | 66.711% | 51.244% | 67.424% | 68.418% | 99.686% | 0.0% | 1.0000 | 6.83e-04 | 9108 |
| E70 | 0.5051 | 0.7701 | 65.438% | 49.812% | 64.893% | 67.590% | 99.647% | 0.0% | 1.0000 | 6.79e-04 | 9240 |

---

## 4. Performance Trajectory & Historical Comparison

| Checkpoint / Milestone | Epoch | Global Step | Val Dice | Val IoU | Val Precision | Val Recall | Val Loss | Zero-Pred % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical Baseline (E56)** | 56 | 7,392 | 65.623% | 49.854% | 69.009% | 63.649% | 0.7639 | 0.0% |
| **Resume Checkpoint (E60)** | 60 | 7,920 | 61.527% | 45.665% | 68.624% | 56.831% | 0.7953 | 0.0% |
| **E61–E70 Best Epoch** | 64 | 8448 | 69.386% | 54.326% | 74.689% | 66.415% | 0.7471 | 0.0% |
| **Final Checkpoint (E70)** | 70 | 9,240 | 65.438% | 49.812% | 64.893% | 67.590% | 0.7701 | 0.0% |

---

## 5. Numerical Stability Audit
- **NaN / Inf Incurrence**: **0 (Zero)**.
- **Teacher/Student BatchNorm Buffers**: Fully synchronized across all 1,320 batches (10 epochs $\times$ 132 batches/epoch).
- **Loss and Gradient Bounds**: Bounded within finite ranges throughout all iterations (Steps 7,921 to 9,240).

---

## 6. Best Validation Checkpoint & Research Conclusion
- **Best Validation Epoch (E56–E70)**: **Epoch 64**
- **Best Validation Dice**: **69.386%**
- **Best Precision**: **74.689%**
- **Best Recall**: **66.415%**
- **Best Validation Loss**: **0.7471**
- **New Validation Best Achieved Beyond E56?**: **YES**
- **Final Model Selection**: Updated to Epoch 64

### Scientific Justification Regarding Further Training
1. **Convergence Behavior**: The model demonstrated marginal gains.
2. **Recommendation**: Model achieved a new peak; evaluate further milestones carefully.
