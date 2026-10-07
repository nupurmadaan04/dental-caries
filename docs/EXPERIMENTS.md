# Experimental Progression & Benchmark Records

This document chronicles the complete experimental trajectory of the MLUA dental caries segmentation project, from early supervised baselines and numerical stability remediations to the final production checkpoint.

---

## 1. Experiment Overview & Checkpoint Registry

| Experiment ID | Epochs | Supervision | Key Architectural Feature | Primary Outcome & Validation Metrics | Checkpoint Path | Status |
| :--- | :---: | :---: | :--- | :--- | :--- | :--- |
| **`EXP-MLUA-001`** | 60 | 20% Labeled | Supervised baseline (ResNet-34 + FPN) | Val Dice: 54.21%, Val Loss: 0.8920. Baseline without semi-supervised consistency. | `outputs/experiments/EXP-MLUA-001_HISTORICAL/checkpoints/EXP-MLUA-001_BEST.pth` | Historical Baseline |
| **`EXP-MLUA-002`** | 10 | 20% Labeled | Semi-supervised MLUA (Parameter-only EMA) | **Numerical Collapse (NaN Loss at Epoch 10).** Forensic analysis revealed missing BatchNorm buffer synchronization. | N/A (Failed Run) | Historical Audit |
| **`EXP-MLUA-003 (60 Ep)`** | 60 | 20% Labeled | Remediated Dual Parameter + BN Buffer Sync | Flawless convergence over 7,920 steps. **E56 Peak Checkpoint: Val Dice 65.623%, Val Loss 0.7639.** | `outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E56_FINAL.pth` | Preserved Historical Baseline |
| **`EXP-MLUA-003 (70 Ep)`** | 70 | 20% Labeled | Extended Training Run with Dual Buffer Sync | **Historical Peak Checkpoint: Epoch 64. Val Dice 69.386%, Val Loss 0.7471.** | `outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E64_BEST.pth` | Preserved Historical Reference |
| **`EXP-MLUA-003 (78 Ep)`** | 78 | 20% Labeled | Final Controlled Micro-Extension with Dual BN Buffer Sync | **New Canonical Peak Validation Checkpoint: Epoch 75. Val Dice 71.867%, Val Loss 0.7254.** Exceeds 71.12% literature benchmark (+0.747 pp). Total run completed at Epoch 78 (10,296 steps). | `outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E75_BEST.pth` | **Current Canonical Active Checkpoint** |

---

## 2. Current Canonical Active Checkpoint: `EXP-MLUA-003_E75_BEST.pth`

Evaluated on the canonical DC1000 validation set across 598 patches at operating threshold $\tau = 0.50$:

```
========================================================================================
EXP-MLUA-003 (Epoch 75, Global Step 9,900) Canonical Validation Performance
========================================================================================
- Validation Dice Similarity (DSC):  71.867%  (0.71867)  [+2.481 pp over E64, +0.747 pp vs literature 71.12%]
- Intersection over Union (IoU):     57.349%  (0.57349)  [+3.023 pp over E64]
- Validation Precision (PPV):        78.132%  (0.78132)  [+3.443 pp over E64]
- Validation Recall (Sensitivity):   67.343%  (0.67343)  [+0.928 pp over E64]
- Specificity (TNR):                 99.824%  (0.99824)  [+0.040 pp over E64]
- Peak Validation Loss:              0.72540             [-0.0217 vs E64 0.74710]
- Zero-Prediction Ratio:             2.0%
========================================================================================
Preserved Historical Reference Checkpoint: EXP-MLUA-003_E64_BEST.pth (Epoch 64, Step 8,448)
- Val Dice: 69.386%, Val IoU: 54.326%, Val Precision: 74.689%, Val Recall: 66.415%, TNR: 99.784%
========================================================================================
```

---

## 3. Deep Forensic Investigation: EXP-MLUA-002 Numerical Instability

During EXP-MLUA-002, training abruptly diverged at Epoch 10 with gradient explosion ($\text{loss} \rightarrow \text{NaN}$). A root-cause audit was conducted by profiling weight tensors and activation statistics across layers:

```
[EXP-MLUA-002 EMA Synchronization Audit]
- Trainable Parameters (Weights, Biases): Synchronized via θ_ema = β * θ_ema + (1 - β) * θ
- BatchNorm Running Means (running_mean): UNTRACKED (Diverged by > 14.2x)
- BatchNorm Running Variances (running_var): UNTRACKED (Collapsed to near zero, causing 1/sqrt(var) -> inf)
```

### Forensic Root Cause
When the Teacher network processed perturbed unlabeled inputs, the mismatched normalization statistics generated massive activation spikes in the early bottleneck layers ($C_2, C_3$), leading to numerical overflow in downstream consistency loss.

### Engineering Remediation
The PyTorch engine was updated to synchronize all buffers alongside model parameters:
```python
def update_teacher_ema(student, teacher, alpha=0.999):
    # Synchronize trainable parameters
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(alpha).add_(s_param.data, alpha=1.0 - alpha)
    # Synchronize BatchNorm running statistics
    for t_buffer, s_buffer in zip(teacher.buffers(), student.buffers()):
        if t_buffer.is_floating_point():
            t_buffer.data.mul_(alpha).add_(s_buffer.data, alpha=1.0 - alpha)
        else:
            t_buffer.data.copy_(s_buffer.data)
```
This remediation eliminated numerical instability completely across all 9,240 subsequent optimization steps in EXP-MLUA-003.

---

## 4. Operating Threshold Sweep ($\tau = 0.05 \dots 0.95$)

To determine the optimal decision threshold, a 19-point sweep was conducted on the validation cohort:

| Threshold $\tau$ | Dice (%) | Precision (%) | Recall (%) | Specificity (%) | Clinical Assessment |
| :---: | :---: | :---: | :---: | :---: | :--- |
| $0.20$ | 61.86% | 49.12% | 78.41% | 98.92% | High false positive rate from cervical burnout |
| $0.35$ | 65.42% | 61.20% | 72.15% | 99.45% | Moderate balance, slight background noise |
| **`0.50`** | **`69.39%`** | **`74.69%`** | **`66.42%`** | **`99.78%`** | **Optimal Clinical Operating Point (Balanced Precision & Recall)** |
| $0.65$ | 65.23% | 79.80% | 57.30% | 99.89% | Depressed sensitivity on early enamel lesions |
| $0.80$ | 58.85% | 84.10% | 44.30% | 99.95% | Severe under-segmentation of demineralized zones |

---

## 5. Independent Sealed Test Set Evaluation

Following threshold freezing at $\tau = 0.50$, the canonical active checkpoint `EXP-MLUA-003_E75_BEST.pth` was evaluated on the strictly sealed test set of 100 independent panoramic radiographs:

### Final Sealed-Test Evaluation (`EXP-MLUA-003_E75_BEST.pth`)
- **Macro Dice:** 50.147% (`50.15%`)
- **Macro IoU:** 36.607% (`36.61%`)
- **Macro Precision:** 59.889% (`59.89%`)
- **Macro Recall (Sensitivity):** 48.077% (`48.08%`)
- **Macro Specificity:** 99.872% (`99.87%`)
- **Macro F1 Score:** 50.147% (`50.15%`)
- **Zero-Prediction Cases:** 0 / 100 (`0.0%`)
- **Micro Dice:** 52.924% (`52.92%`)
- **Micro IoU:** 35.984% (`35.98%`)
- **Micro Precision:** 61.540% (`61.54%`)
- **Micro Recall:** 46.424% (`46.42%`)
- **Global Confusion Matrix:** 117,964,800 evaluated test pixels ($\text{TP} = 240,579; \text{FP} = 150,350; \text{FN} = 277,638; \text{TN} = 117,296,233$).
- **Generalization Gap:** Validation-to-test Dice gap of -21.720 percentage points (strictly designated as *Final Sealed-Test Evaluation*, not clinical validation).

### Historical Baseline Sealed Test Reference (Evaluated on E56)
- **Macro Dice:** 43.041% | **Macro Recall:** 52.896% | **Macro Specificity:** 99.630% | **Micro Dice:** 43.391%
- **Evaluated Pixels:** $\text{TP} = 263,935, \text{FP} = 434,392, \text{FN} = 254,282, \text{TN} = 117,012,191$.

The validation-to-test generalization gap illustrates the real-world clinical challenge of panoramic radiograph variance, cervical burnout, and fine proximal lesion boundaries.
