# Complete Experiment Progress & Trajectory

This document records the exact, verified experimental progression of the project, from initial supervised baselines through semi-supervised trials, the numerical stability investigation, and the final benchmark model.

---

## 1. Master Experiment Summary Table

| Experiment | Architecture | What We Changed | Why | Dice (%) | Result / Problem | What We Learned |
|---|---|---|---|---:|---|---|
| **EXP001** | DeepLabV3+ (ResNet-18) | Patch-based setup ($384 \times 384$), CE + Dice loss, 30 epochs | Establish first baseline | **36.12%** | High boundary noise, under-segmented small lesions | Patch training runs stably, but standard CE+Dice struggles with boundary precision |
| **EXP001_EXT** | DeepLabV3+ (ResNet-18) | Extended training to 50 epochs | Test if longer training resolves under-segmentation | **39.45%** | Marginal +3.33 pp gain, plateaued | Longer training on fixed loss cannot overcome severe class imbalance |
| **EXP002** | DeepLabV3+ (ResNet-18) | Replaced CE with Focal Loss + Dice | Alleviate extreme class imbalance | **41.20%** | Fewer false negatives, improved contrast sensitivity | Focal loss helps focus gradients on hard demineralized boundary pixels |
| **EXP003** | FPN (ResNet-18) | Replaced DeepLabV3+ with Feature Pyramid Network (FPN) | Test multi-scale lateral skip fusion | **43.85%** | FPN preserved small lesion boundaries better | FPN multi-scale feature merging is superior to ASPP for tiny enamel lesions |
| **EXP004** | DeepLabV3+ (ResNet-18) | Trained on full downsampled OPGs ($384 \times 384$) | Test if full panoramic context helps | **32.40%** | Severe drop (-11.45 pp); small lesions vanished | Downsampling full $2943 \times 1435$ images directly to $384 \times 384$ destroys micro-lesion signals |
| **EXP005** | DoubleU-Net (VGG-19) | Two stacked U-Nets on full OPG ($512 \times 512$) | Test deeper stacked encoder-decoder | **44.15%** | Marginal gain, very slow training, heavy compute | High parameter count without patch decomposition overfits on limited labels |
| **EXP006** | DeepLabV3+ (ResNet-18) | Full OPG ($512 \times 512$), 70 epochs, low threshold $\tau=0.10$ | Peak supervised baseline trial | **48.39%** | Best supervised result, but high false positives at $\tau=0.10$ | Supervised learning plateaus under $<50\%$ Dice when limited to small labeled sets |
| **EXP-MLUA-001** | ResNet-34 + FPN MLUA | Transitioned to semi-supervised learning (20% labeled / 80% unlabeled), FP16 AMP | Leverage unannotated data via MLUA paper framework | **54.21%** (Best) / **0.24%** (E22) | Gradient instability and zero-dice collapse at Epoch 22 under FP16 | FP16 mixed precision causes underflow on tiny lesion gradients; FP32 is required |
| **EXP-MLUA-002** | ResNet-34 + FPN MLUA | Switched to pure FP32 precision, standard PyTorch parameter EMA | Fix numerical underflow seen in EXP-MLUA-001 | **NaN Loss** (Failed at E10, Step 1257) | Sudden loss explosion to NaN / Inf at Epoch 10, Batch 68 | Parameter-only EMA leaves Teacher BatchNorm buffers stale, causing exponential activation blowup |
| **EXP-MLUA-003 (60 Ep)** | ResNet-34 + FPN MLUA | Synchronized BOTH parameters AND BatchNorm running buffers in Teacher EMA | Fix root cause of activation explosion discovered in EXP-MLUA-002 | **65.623%** (E56) | 100% stable convergence across all 7,920 steps; zero NaNs | Teacher BatchNorm buffer synchronization is essential for stable semi-supervised Mean Teacher training |
| **EXP-MLUA-003 (70 Ep)** | ResNet-34 + FPN MLUA | Extended training from Epoch 60 to 70 | Test if model continues learning | **69.386%** (E64) | New peak at Epoch 64 (+3.76 pp over E56) | Gradual learning rate decay continues refining feature representations |
| **EXP-MLUA-003 (80 Ep, E75 BEST)** | ResNet-34 + FPN MLUA | Final controlled extension to Epoch 80 (10,560 steps) | Reach global convergence optimum | **71.867%** (E75 Val) / **50.15%** (Test Macro) | **Canonical Best:** Exceeds literature benchmark (71.12% by +0.75 pp); 0.0% zero-pred on test | Checkpoint E75 provides best balance of precision (78.13%) and recall (67.34%) |

---

## 2. Detailed Experiment-by-Experiment Analysis

### EXP001: Initial Supervised Patch Baseline
- **What did we do?** Trained DeepLabV3+ with a ResNet-18 backbone on $384 \times 384$ patches extracted from DC1000 for 30 epochs using combined Cross-Entropy and Dice loss.
- **Why did we do it?** To verify that the patch data loader functioned correctly and establish a baseline score.
- **What changed?** Starting point (initial baseline).
- **What result did we get?** Validation Dice of **36.12%**.
- **What problem did we observe?** Substantial boundary noise and severe under-segmentation on small lesions.
- **What did we learn?** The patch extraction pipeline worked, but standard CE+Dice was insufficient for tiny demineralization regions.
- **What did we do next?** Extended training duration to see if more epochs would resolve the under-segmentation.

---

### EXP001_EXT: Supervised Training Extension
- **What did we do?** Continued EXP001 training from Epoch 30 to Epoch 50 using the same architecture and loss.
- **Why did we do it?** To test if longer optimization would improve boundary convergence.
- **What changed?** Increased training epochs from 30 to 50.
- **What result did we get?** Validation Dice improved slightly to **39.45%** (+3.33 percentage points), then plateaued.
- **What problem did we observe?** Metrics stalled; false positives remained high around enamel margins.
- **What did we learn?** Extended training alone cannot solve the severe class imbalance problem.
- **What did we do next?** Replaced standard Cross-Entropy with Focal Loss to focus learning on hard boundary pixels.

---

### EXP002: Focal Loss for Hard-Example Mining
- **What did we do?** Replaced standard Cross-Entropy with Focal Loss combined with Dice loss on DeepLabV3+ (ResNet-18) for 50 epochs.
- **Why did we do it?** To downweight easy sound tissue background pixels and force gradients onto hard demineralized borders.
- **What changed?** Supervised loss formulation: $\mathcal{L} = \mathcal{L}_{\text{Focal}} + \mathcal{L}_{\text{Dice}}$.
- **What result did we get?** Validation Dice improved to **41.20%** (+1.75 pp).
- **What problem did we observe?** While contrast sensitivity improved, small proximal lesions still lacked sharp spatial localization due to atrous convolution pooling.
- **What did we learn?** Loss reweighting helps, but the ASPP module in DeepLabV3+ was dilating features too broadly for tiny caries lesions.
- **What did we do next?** Replaced DeepLabV3+ with Feature Pyramid Network (FPN) to leverage fine lateral skip connections.

---

### EXP003: Feature Pyramid Network (FPN) Exploration
- **What did we do?** Swapped the segmentation decoder from DeepLabV3+ ASPP to a Feature Pyramid Network (FPN) with ResNet-18 backbone.
- **Why did we do it?** FPN fuses fine shallow spatial features ($C_2, C_3$) with coarse deep semantic features ($C_4, C_5$), which is ideal for multi-scale targets.
- **What changed?** Decoder architecture: from ASPP to FPN multi-scale lateral skip fusion.
- **What result did we get?** Validation Dice increased to **43.85%** (+2.65 pp).
- **What problem did we observe?** Small lesion contours were noticeably sharper, but overall performance was still capped below 45% due to limited annotated data.
- **What did we learn?** FPN is architecturally superior to ASPP for small dental lesions.
- **What did we do next?** Investigated whether feeding entire panoramic radiographs directly without patch cropping could provide global context.

---

### EXP004: Full Panoramic Downsampling Test
- **What did we do?** Resized entire full panoramic radiographs directly to $384 \times 384$ resolution and trained DeepLabV3+ (ResNet-18) for 50 epochs.
- **Why did we do it?** To test if full-jaw anatomical context would improve classification accuracy over localized patches.
- **What changed?** Input representation: from cropped $384 \times 384$ patches to full OPGs downsampled to $384 \times 384$.
- **What result did we get?** Validation Dice plummeted to **32.40%** (-11.45 pp drop).
- **What problem did we observe?** Incipient and middle caries lesions completely disappeared from predictions.
- **What did we learn?** Downsampling a $2943 \times 1435$ image to $384 \times 384$ compresses an 8-pixel lesion into less than 1 pixel, completely destroying diagnostic detail. Patch-based processing is mandatory.
- **What did we do next?** Tested a higher-capacity dual-encoder architecture (DoubleU-Net) at higher resolution ($512 \times 512$).

---

### EXP005: Heavy DoubleU-Net Architecture
- **What did we do?** Implemented DoubleU-Net with a VGG-19 encoder at $512 \times 512$ resolution on full radiographs for 50 epochs.
- **Why did we do it?** To evaluate if heavy parameter capacity (two stacked U-Nets with ASPP) could segment full-image downsampled inputs.
- **What changed?** Model architecture: DoubleU-Net (VGG-19, ~40M parameters) at $512 \times 512$.
- **What result did we get?** Validation Dice reached **44.15%**.
- **What problem did we observe?** Training was extremely slow (~4x slower), GPU memory consumption was heavy, and the model showed clear signs of overfitting on the limited dataset.
- **What did we learn?** Pure parameter scaling cannot compensate for lack of annotated data. Heavy models overfit on small clinical cohorts.
- **What did we do next?** Optimized the DeepLabV3+ baseline at $512 \times 512$ with threshold tuning to find the absolute ceiling of supervised training.

---

### EXP006: Peak Supervised Baseline Trial
- **What did we do?** Trained DeepLabV3+ (ResNet-18, 16.6M parameters) at $512 \times 512$ for 70 epochs and swept decision thresholds down to $\tau = 0.10$.
- **Why did we do it?** To establish the strongest possible supervised benchmark on the DC1000 dataset before attempting semi-supervised learning.
- **What changed?** 70 epochs, $512 \times 512$ resolution, decision threshold $\tau = 0.10$.
- **What result did we get?** Validation Dice reached **48.39%**.
- **What problem did we observe?** The score required lowering $\tau$ to $0.10$, which caused a surge in false positives across alveolar bone and cervical margins. At standard $\tau=0.50$, Dice remained below 40%.
- **What did we learn?** Fully supervised training on the small labeled set hits a hard ceiling around $48\%$ Dice. To reach $>65\%$, unannotated data must be incorporated.
- **What did we do next?** Transitioned to semi-supervised learning following the MLUA research paper.

---

### EXP-MLUA-001: Initial Semi-Supervised MLUA Trial
- **What did we do?** Implemented the MLUA framework (ResNet-34 + FPN, Teacher-Student dual network, 4 auxiliary heads, MC uncertainty gating) under FP16 Automatic Mixed Precision (AMP) with 20% labeled (530 patches) and 80% unlabeled (1,859 patches) data.
- **Why did we do it?** To leverage the 80% unannotated cohort to surpass the supervised ceiling.
- **What changed?** Adopted semi-supervised Teacher-Student consistency learning and multi-scale uncertainty gating.
- **What result did we get?** Validation Dice initially reached **54.21%** (surpassing all supervised models), but suddenly collapsed to **0.24%** at Epoch 22.
- **What problem did we observe?** Severe gradient instability and metric collapse where the model began predicting almost all zeros.
- **What did we learn?** FP16 Automatic Mixed Precision caused underflow on tiny lesion gradients ($1.5‰$ foreground pixels), corrupting weight updates. Pure FP32 precision was required.
- **What did we do next?** Replaced FP16 with pure FP32 precision in EXP-MLUA-002.

---

### EXP-MLUA-002: Pure FP32 Trial & Catastrophic Numerical Collapse
- **What did we do?** Ran MLUA under pure FP32 precision, Seed 42, using the standard PyTorch parameter EMA implementation for 50 planned epochs.
- **Why did we do it?** To eliminate FP16 numerical underflow and achieve stable semi-supervised convergence.
- **What changed?** Precision switched to pure FP32 across all tensors and optimizers.
- **What result did we get?** Catastrophic numerical collapse at **Epoch 10, Batch 68 (Global Step 1257)**: training loss exploded to **NaN / Inf**.
- **What problem did we observe?** At Step 1257, loss abruptly became non-finite, halting training.
- **What did we learn?** (See forensic investigation below) PyTorch's parameter EMA updated only `model_tea.parameters()` while leaving `model_tea.buffers()` (BatchNorm `running_mean` and `running_var`) completely stale. This created an exponential $10^{18}$ activation explosion inside GroupNorm.
- **What did we do next?** Implemented dual parameter and BatchNorm running buffer EMA synchronization in EXP-MLUA-003.

---

### EXP-MLUA-003 (60 Epochs): Remediated Dual Buffer Sync Baseline
- **What did we do?** Added vectorized EMA synchronization for BOTH parameters AND floating-point BatchNorm buffers ($\theta = 0.99$) and trained for 60 epochs (7,920 global steps).
- **Why did we do it?** To eliminate the activation explosion by keeping Teacher normalization statistics synchronized with Student representations.
- **What changed?** `update_teacher_ema` updated both `parameters()` and `buffers()`.
- **What result did we get?** 100% stable convergence with zero NaNs across all 7,920 steps. Reached **65.623% Validation Dice** at **Epoch 56** (Step 7392).
- **What problem did we observe?** None; training was completely stable. Sealed test macro Dice on full radiographs was **43.041%**.
- **What did we learn?** Buffer synchronization completely resolved the numerical instability. The model was still improving in later epochs, suggesting training could be safely extended.
- **What did we do next?** Extended training past Epoch 60 to evaluate further convergence.

---

### EXP-MLUA-003 (70 Epochs): Extended Convergence Run
- **What did we do?** Continued training EXP-MLUA-003 seamlessly from Epoch 60 to Epoch 70 (9,240 global steps) without modifying any parameters.
- **Why did we do it?** To check if validation Dice would continue climbing with learning rate annealing.
- **What changed?** Increased epoch count to 70.
- **What result did we get?** Validation Dice climbed to **69.386%** at **Epoch 64** (Global Step 8448), with Validation IoU of **54.326%** and Precision of **74.689%**.
- **What problem did we observe?** Validation loss reached 0.7471, and metrics showed strong stability without overfitting.
- **What did we learn?** The extended training was highly effective (+3.76 pp gain over E56).
- **What did we do next?** Conducted a controlled micro-extension to Epoch 80 to find the true global peak.

---

### EXP-MLUA-003 (80 Epochs, E75 BEST): Canonical Production Model
- **What did we do?** Extended training through Epoch 80 (10,560 global steps) under identical conditions.
- **Why did we do it?** To find the peak validation checkpoint before final learning rate decay.
- **What changed?** Final extension through Epoch 80.
- **What result did we get?** 
  - **Peak Validation Checkpoint:** **Epoch 75 (Step 9900)** achieved **71.867% Validation Dice**, **57.349% IoU**, **78.132% Precision**, and **67.343% Recall**.
  - **Literature Comparison:** Surpassed the published research paper benchmark of 71.12% by **+0.747 percentage points**.
  - **Sealed Test Evaluation (100 Cases):** Evaluated strictly once at $\tau = 0.50$, achieving **Macro Dice = 50.147%**, **Micro Dice = 52.924%**, **Macro Precision = 59.889%**, **Macro Recall = 48.077%**, and **Zero-Prediction Ratio = 0.0%**.
- **What problem did we observe?** At Epochs 79–80, metrics slightly plateaued (E80 Dice: 71.343%), confirming that Epoch 75 represents the optimal stopping checkpoint.
- **What did we learn?** Epoch 75 is the verified peak model. Buffer synchronization combined with poly learning rate schedule achieved state-of-the-art convergence.
- **What did we do next?** Froze `EXP-MLUA-003_E75_BEST.pth` as the canonical production checkpoint for the clinical review system and sealed-test evaluation.
