# Failed Experiments, Forensic Analysis, and Engineering Fixes

This document records the exact failures encountered during the project, the forensic evidence collected, the identified root causes, the engineering fixes applied, and the verified results after remediation.

---

## 1. Summary of Major Experimental Failures & Remediations

| Experiment | What Failed | Evidence | Root Cause | Fix Applied | Result After Fix |
|---|---|---|---|---|---|
| **EXP004** | Loss of small lesions (-11.45 pp Dice drop) | `val_dice = 32.40%` vs EXP003 `43.85%` | Direct downsampling of full OPGs ($2943 \times 1435$) to $384 \times 384$ compresses lesions $<8\text{ px}$ into sub-pixel space | Abandoned direct downsampling; adopted $384 \times 384$ overlapping patch extraction (21 patches/OPG) | Lesion pixel density restored; baseline improved to $>43\%$ |
| **EXP005** | Overfitting and unviable compute footprint | Training time $\approx 4\times$ longer, plateaued at $44.15\%$ Dice | DoubleU-Net (~40M parameters) is excessively heavy for 530 labeled patches; lacked multi-scale uncertainty gating | Replaced heavy DoubleU-Net with ResNet-34 + FPN (~21M parameters) | Faster training, stable capacity, well-matched to dataset size |
| **EXP-MLUA-001** | Gradient instability & zero-prediction collapse at Epoch 22 | Validation Dice collapsed from $14.71\%$ (E19) to $0.24\%$ (E22) | FP16 Automatic Mixed Precision (AMP) caused numerical underflow on tiny lesion gradients ($1.5‰$ foreground) | Replaced FP16 AMP with pure FP32 precision across all network layers and optimizer buffers | Underflow eliminated; initial training ran stably |
| **EXP-MLUA-002** | Catastrophic loss explosion ($\text{Loss} \rightarrow \text{NaN}$) at Epoch 10, Batch 68 | Training halted at Global Step 1257; `batch68_deep_dive_log.txt` showed NaNs in GroupNorm | Teacher parameter EMA only updated `parameters()`, leaving BatchNorm `buffers()` (`running_mean`, `running_var`) stale and un-updated | Vectorized EMA synchronization updated BOTH `parameters()` AND floating-point `buffers()` in `src/mlua/engine/` | **EXP-MLUA-003 achieved 100% finite convergence across 10,560 steps with ZERO NaNs**, reaching 71.87% Dice |

---

## 2. Deep Forensic Investigation: The EXP-MLUA-002 Numerical Collapse

### 1. What Happened?
At **Epoch 10, Batch 68 (Global Step 1257)**, training abruptly halted when the loss function output became non-finite (`NaN` / `Inf`).

### 2. At Which Stage Did the Failure Occur?
The failure occurred inside the **Teacher Network forward pass on perturbed unlabeled images** during consistency loss calculation. Specifically, the first non-finite activation was traced to:
`model_tea.decoder.seg_blocks.0.block.0.block.1` (a GroupNorm layer operating on feature map shape $[36, 128, 12, 12]$).

### 3. What Evidence Did We Collect?
We instrumented the PyTorch model with forward hooks to trace layer-by-layer activation magnitudes leading up to Batch 68 (`outputs/diagnostics/EXP-MLUA-002_E10_ACTIVATION_FORENSIC/activation_forensic_log.txt`):

```text
Layer Trace Leading to Failure at Batch 68 (Unlabeled Image 3):
- Input Tensor:              abs_max = 1.0038e+00  (Finite)
- encoder.conv1:             abs_max = 2.1136e+00  (Finite)
- encoder.bn1:               abs_max = 2.0990e+00  (Finite)
- encoder.layer1 (c2):       abs_max = 3.4572e+02  (Growth begins)
- encoder.layer2 (c3):       abs_max = 1.1084e+07  (Rapid scaling)
- encoder.layer3 (c4):       abs_max = 1.8997e+15  (Massive explosion)
- encoder.layer4 (c5):       abs_max = 9.4190e+18  (Extremely large)
- decoder.p5:                abs_max = 1.4907e+19  (Extremely large)
- decoder.seg_blocks GroupNorm: (1.49e+19)^2 ≈ 2.2e+38 -> EXCEEDS FP32 MAX (3.4e+38) -> OVERFLOW -> NaN
```

### 4. What Was the Actual Root Cause?
- In PyTorch, standard Exponential Moving Average (EMA) implementations iterate exclusively over `model.parameters()`:
  ```python
  # Standard (flawed) parameter EMA:
  for t_param, s_param in zip(teacher.parameters(), student.parameters()):
      t_param.data.mul_(alpha).add_(s_param.data, alpha=1.0 - alpha)
  ```
- **The flaw:** Batch normalization layers store their running statistics (`running_mean` and `running_var`) in `model.buffers()`, NOT `parameters()`.
- Because buffers were never updated, the Teacher network retained stale normalization statistics from early initialization (Epoch 1).
- As the Student weights evolved through 1,257 gradient steps, the discrepancy between the Teacher's evolving weights and its static, stale BatchNorm statistics caused input signals to be scaled up by an order of magnitude at every successive convolutional stage ($10^2 \to 10^7 \to 10^{15} \to 10^{18}$).
- When activations reached GroupNorm, computing internal channel variance squared $1.49 \times 10^{19}$, yielding $\approx 2.2 \times 10^{38}$. This hit the IEEE 754 single-precision float ceiling ($\approx 3.4 \times 10^{38}$), causing arithmetic overflow to `+inf`, which immediately resulted in `NaN` variance.

### 5. What Was NOT the Cause?
- **It was NOT an exploding learning rate:** The learning rate was small ($0.00095$ under AdamW with polynomial decay).
- **It was NOT corrupted input data:** All input image tensors and masks were verified finite and bounded in $[0.0, 1.0]$.
- **It was NOT an architectural flaw in MLUA:** The MLUA formulation itself is mathematically sound.
- **It was NOT gradient clipping failure:** Gradients had not yet backpropagated from the Teacher (the Teacher is updated via EMA, not gradients).

### 6. What Code Correction Was Made?
In `src/mlua/engine/trainer.py` (and `train_exp003.py`), we rewrote the Teacher EMA update function to synchronize both parameters AND floating-point buffers:

```python
def update_teacher_ema(student_model, teacher_model, alpha=0.999):
    # 1. Synchronize trainable weights & biases
    for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
        t_param.data.mul_(alpha).add_(s_param.data, alpha=1.0 - alpha)
        
    # 2. Synchronize BatchNorm running buffers (running_mean, running_var)
    for t_buffer, s_buffer in zip(teacher_model.buffers(), student_model.buffers()):
        if t_buffer.is_floating_point():
            t_buffer.data.mul_(alpha).add_(s_buffer.data, alpha=1.0 - alpha)
        else:
            t_buffer.data.copy_(s_buffer.data)
```

### 7. What Happened Afterward?
In **EXP-MLUA-003**, the counterfactual buffer synchronization was validated:
- **10,560 optimization steps (80 epochs) completed with 100% finite numbers.**
- **Zero NaNs, zero Infs, and zero activation blowups.**
- Teacher activations remained stably bounded ($\text{abs\_max} \le 12.0$).
- The model trained cleanly to achieve **71.867% validation Dice at Epoch 75**.

---

## 3. Clear Categorical Distinctions for Mentor Viva

| Distinction | Description |
|---|---|
| **Research Paper Method** | The Mean Teacher semi-supervised paradigm, FPN multi-scale supervision, and Monte Carlo uncertainty gating proposed by Wang et al. (2023). |
| **Our Implementation Bug** | Omitting BatchNorm `running_mean` and `running_var` buffer tracking inside the PyTorch EMA update loop during EXP-MLUA-002. |
| **Our Engineering Fix** | Extending the EMA update routine to apply dual parameter and floating-point buffer synchronization at every training batch. |
