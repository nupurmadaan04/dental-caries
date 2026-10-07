# Improvement Story: Progression Chains & Empirical Learnings

This document highlights the major turning points in our research where an observed limitation or failure led to a specific engineering intervention, resulting in measurable empirical improvement.

---

## Improvement Chain 1: Supervised Patch Baseline → Loss Function Optimization

```text
Previous:
EXP001 (DeepLabV3+ with Cross-Entropy + Dice) achieved 36.12% Validation Dice.

Problem:
Severe foreground/background class imbalance (caries occupies <0.15% on full OPGs) caused standard cross-entropy gradients to be overwhelmed by sound enamel and background pixels, leading to high false-negative rates on subtle lesions.

Change:
In EXP002, replaced standard Cross-Entropy with Focal Loss (gamma = 2.0) combined with Dice loss.

Reason:
Focal loss dynamically downweights well-classified background pixels, focusing gradient updates on difficult, ambiguous lesion boundary pixels.

New Result:
Validation Dice increased to 41.20% (+5.08 percentage points over EXP001).

Learning:
Loss formulation must explicitly address medical class imbalance to prevent foreground gradient vanishing.
```

---

## Improvement Chain 2: ASPP Dilated Convolutions → FPN Multi-Scale Lateral Fusion

```text
Previous:
EXP002 (DeepLabV3+ with ASPP decoder) achieved 41.20% Validation Dice.

Problem:
Atrous Spatial Pyramid Pooling (ASPP) applies large dilation rates (6, 12, 18) that smooth out and dilute fine spatial boundaries. Incipient enamel lesions (<100 pixels) lost sharp boundary delineation.

Change:
In EXP003, replaced the DeepLabV3+ ASPP decoder with a Feature Pyramid Network (FPN) decoder with lateral skip connections.

Reason:
FPN directly connects high-resolution shallow feature maps (stages C2, C3) with deep semantic feature maps (stages C4, C5), preserving microscopic spatial coordinates alongside contextual semantics.

New Result:
Validation Dice improved to 43.85% (+2.65 percentage points over EXP002).

Learning:
Multi-scale pyramid feature fusion is structurally superior to atrous pooling for tiny, irregular medical lesions.
```

---

## Improvement Chain 3: Supervised Learning Ceiling → Semi-Supervised MLUA Paradigm

```text
Previous:
EXP006 achieved peak supervised Validation Dice of 48.39% at 512x512 resolution (tau = 0.10).

Problem:
Fully supervised training hit an empirical ceiling below 50% Dice because only 530 labeled patches were available. Training longer or adding model parameters (DoubleU-Net, 40M params) caused overfitting without accuracy gains.

Change:
Adopted the semi-supervised Multi-Level Uncertainty-Aware (MLUA) framework, using a Teacher-Student Mean Teacher architecture to leverage the 1,859 unannotated patches (80% of cohort) with Monte Carlo uncertainty gating.

Reason:
Semi-supervised consistency regularization forces the network to learn rich, domain-invariant dental tooth representations from unannotated clinical images, multiplying available training data by 4.5x.

New Result:
Initial semi-supervised validation reached 54.21% in EXP-MLUA-001, decisively surpassing the supervised ceiling of 48.39%.

Learning:
When clinical annotations are scarce, semi-supervised consistency learning is far more effective than increasing model parameter capacity.
```

---

## Improvement Chain 4: Numerical Instability & NaN Loss → Dual BatchNorm Buffer Synchronization

```text
Previous:
EXP-MLUA-002 collapsed at Epoch 10, Batch 68 (Global Step 1257) with loss exploding to NaN / Inf.

Problem:
PyTorch's standard Parameter EMA implementation updated only model_tea.parameters(). The Teacher's BatchNorm running buffers (running_mean and running_var) were left static from initialization. As Student weights evolved, the Teacher evaluated inputs using stale normalization buffers, causing intermediate activations to scale exponentially (10^2 -> 10^7 -> 10^15 -> 10^18) until hitting GroupNorm arithmetic overflow (> 3.4e+38 in float32).

Change:
In EXP-MLUA-003, rewrote update_teacher_ema to vectorize Exponential Moving Average updates across BOTH trainable parameters AND floating-point BatchNorm buffers:
`t_buffer.data.mul_(alpha).add_(s_buffer.data, alpha=1.0 - alpha)`.

Reason:
Keeping Teacher normalization statistics tightly aligned with Student parameter distributions ensures that activations remain normalized and bounded throughout the entire forward pass.

New Result:
EXP-MLUA-003 achieved 100% finite, smooth convergence across all 10,560 steps (80 epochs) with zero NaNs, reaching 65.62% Validation Dice at Epoch 56, 69.39% at Epoch 64, and 71.87% at Epoch 75.

Learning:
In semi-supervised Teacher-Student networks, Exponential Moving Average synchronization must encompass both parameters and relevant running buffers.
```

---

## Improvement Chain 5: Milestone Checkpoints (E56 → E64 → E75)

```text
Previous:
Historical baseline checkpoint E56 achieved 65.623% Validation Dice.

Problem:
Training loss and validation loss were still steadily decreasing at Epoch 56, indicating that the model had not yet converged to its global minimum.

Change:
Performed controlled training extensions from Epoch 60 to Epoch 70, and subsequently from Epoch 71 to Epoch 80, maintaining 100% identical hyperparameters and random seed (Seed 42).

Reason:
Polynomial learning rate annealing allowed the optimizer to refine boundary feature representations without destabilizing learned weights.

New Result:
- Epoch 64 achieved 69.386% Validation Dice (+3.76 pp over E56).
- Epoch 75 achieved 71.867% Validation Dice (+2.48 pp over E64, +6.24 pp over E56), surpassing the 71.12% literature benchmark.
- Sealed test Macro Dice on full radiographs improved from 43.041% (E56) to 50.147% (E75, +7.11 pp gain).

Learning:
Controlled training continuation under verified numerical stability and appropriate learning rate scheduling successfully reached peak model convergence.
```
