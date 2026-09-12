# EXP-MLUA-001 vs. EXP-MLUA-002 CONFIGURATION & CODE-PATH AUDIT
**Document**: `docs/EXP001_VS_EXP002_CONFIGURATION_AUDIT.md`  
**Audit Scope**: Strict Forensic Verification of Single-Variable Invariance (DICE265 vs. DICE530 Setting)  
**Status**: **`PASS` — VERIFIED: THE ONLY SCIENTIFIC DIFFERENCE IS THE LABELED/UNLABELED PARTITION (265/2124 $\to$ 530/1859)**

---

## 1. Executive Summary & Verification Matrix

This audit strictly validates that **EXP-MLUA-002** differs from **EXP-MLUA-001** in **exactly one deliberate scientific variable**: the labeled sample count increasing from **265 labeled slices/patches (10% SSL)** to **530 labeled slices/patches (20% SSL)**, with the remaining pool allocated to the unlabeled set ($2,124 \to 1,859$).

All other architecture, optimization, regularization, perturbation, uncertainty aggregation, validation, and evaluation geometries remain 100% strictly invariant.

---

## 2. 25-Point Configuration Comparison Table

| # | Parameter | EXP-MLUA-001 (10% SSL) | EXP-MLUA-002 (20% SSL / DICE530) | Difference | Expected? |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **1** | **Labeled Sample Count** | **265 Patches** (Indices `0..264`) | **530 Patches** (Indices `0..529`) | **YES (Deliberate)** | **YES (Primary Experimental Target)** |
| **2** | **Unlabeled Sample Count** | **2,124 Patches** (Indices `265..2388`) | **1,859 Patches** (Indices `530..2388`) | **YES (Deliberate)** | **YES (Complementary Pool)** |
| **3** | **Random Seed** | `seed = 42` | `seed = 42` | **NO** | **YES (Strict Invariance)** |
| **4** | **Encoder Backbone** | ResNet-34 (`resnet34`, ImageNet init) | ResNet-34 (`resnet34`, ImageNet init) | **NO** | **YES (Strict Invariance)** |
| **5** | **Decoder Channels** | FPN ($P_{\text{channels}}=256, S_{\text{channels}}=128$) | FPN ($P_{\text{channels}}=256, S_{\text{channels}}=128$) | **NO** | **YES (Strict Invariance)** |
| **6** | **Auxiliary Decoder Heads** | 4 Heads + 1 Fused Head (5 Total) | 4 Heads + 1 Fused Head (5 Total) | **NO** | **YES (Strict Invariance)** |
| **7** | **Total SSL Batch Size** | 8 | 8 | **NO** | **YES (Strict Invariance)** |
| **8** | **Labeled Batch Stream** | 4 | 4 | **NO** | **YES (Strict Invariance)** |
| **9** | **Unlabeled Batch Stream** | 4 | 4 | **NO** | **YES (Strict Invariance)** |
| **10**| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$) | AdamW ($\beta_1=0.9, \beta_2=0.999$) | **NO** | **YES (Strict Invariance)** |
| **11**| **Learning Rate** | $\text{lr} = 0.001$ ($10^{-3}$) | $\text{lr} = 0.001$ ($10^{-3}$) | **NO** | **YES (Strict Invariance)** |
| **12**| **Weight Decay** | $\text{wd} = 0.01$ ($10^{-2}$) | $\text{wd} = 0.01$ ($10^{-2}$) | **NO** | **YES (Strict Invariance)** |
| **13**| **LR Scheduler** | Polynomial Decay ($\text{power}=0.9, E=200$) | Polynomial Decay ($\text{power}=0.9, E=200$) | **NO** | **YES (Strict Invariance)** |
| **14**| **Teacher EMA Theta** | $\alpha = 0.99$ | $\alpha = 0.99$ | **NO** | **YES (Strict Invariance)** |
| **15**| **MC Iterations ($T$)** | $T = 8$ Stochastic Passes | $T = 8$ Stochastic Passes | **NO** | **YES (Strict Invariance)** |
| **16**| **Gaussian Perturbation**| $\sigma=0.01$, clamp $[-0.1, 0.1]$ | $\sigma=0.01$, clamp $[-0.1, 0.1]$ | **NO** | **YES (Strict Invariance)** |
| **17**| **Dynamic Uncertainty Threshold** | $E_{\text{th}} = [0.75(1-\frac{e}{E}) + 1.0(\frac{e}{E})]\ln(2)$ | $E_{\text{th}} = [0.75(1-\frac{e}{E}) + 1.0(\frac{e}{E})]\ln(2)$ | **NO** | **YES (Strict Invariance)** |
| **18**| **Consistency Loss Weight** | Sigmoid ramp-up to $w_{\max}=0.10$ | Sigmoid ramp-up to $w_{\max}=0.10$ | **NO** | **YES (Strict Invariance)** |
| **19**| **Supervised Loss Form** | BCE + Dice with 4-level deep supervision | BCE + Dice with 4-level deep supervision | **NO** | **YES (Strict Invariance)** |
| **20**| **Patch Size & Geometry**| $384 \times 384$ pixels (Grayscale 1-channel) | $384 \times 384$ pixels (Grayscale 1-channel) | **NO** | **YES (Strict Invariance)** |
| **21**| **Validation Protocol** | Deterministic unaugmented sliding window | Deterministic unaugmented sliding window | **NO** | **YES (Strict Invariance)** |
| **22**| **Decision Threshold** | Fixed at $\tau = 0.50$ | Fixed at $\tau = 0.50$ | **NO** | **YES (Strict Invariance)** |
| **23**| **Panoramic Reconstruction**| $768 \times 1536$ Canvas (21 patches, stride 192) | $768 \times 1536$ Canvas (21 patches, stride 192) | **NO** | **YES (Strict Invariance)** |
| **24**| **Output & Checkpoint Isolation**| `outputs/experiments/EXP-MLUA-001_HISTORICAL/` | `outputs/experiments/EXP-MLUA-002_HISTORICAL/` | **YES (Isolated)** | **YES (Safety Guarantee)** |
| **25**| **Sealed Benchmark Safety**| `dataset/test/` (100 cases sealed) | `dataset/test/` (100 cases sealed) | **NO** | **YES (Protected & Untouched)** |

---

## 3. Code-Path & Execution Traceability

1. **Data Loader Partitioning**:
   - `EXP-MLUA-001`: `active_rate: "0.1"` $\to$ `labeled_indices = indices[:265]`, `unlabeled_indices = indices[265:]`.
   - `EXP-MLUA-002`: `active_rate: "0.2"` $\to$ `labeled_indices = indices[:530]`, `unlabeled_indices = indices[530:]`.
   - Both use identical `TwoStreamBatchSampler` batching logic yielding $4\text{ labeled} + 4\text{ unlabeled}$ per batch.
2. **Model Graph & Tensor Execution**:
   - Both construct the exact same `Net(in_c=1, out_c=1, encoder_name="resnet34")` with 5 output heads (1 fused head + 4 auxiliary heads).
3. **Teacher Forward & Perturbation**:
   - Both utilize the verified Day-2 **Unified 36-Image Teacher Forward Pass** ($4\text{ base} + 4 \times 8\text{ MC} = 36\text{ images}$) under `torch.inference_mode()`.
4. **Namespace & Checkpoint Isolation**:
   - `EXP-MLUA-001` checkpoints write strictly to `outputs/experiments/EXP-MLUA-001_HISTORICAL/checkpoints/EXP-MLUA-001_LATEST.pth`.
   - `EXP-MLUA-002` checkpoints write strictly to `outputs/experiments/EXP-MLUA-002_HISTORICAL/checkpoints/EXP-MLUA-002_LATEST.pth`.
   - Zero path collisions exist between the two experiment namespaces.

---

## 4. Final Scientific Audit Confirmation

- **Scientific Invariance**: The **ONLY** difference between EXP-MLUA-001 and EXP-MLUA-002 is the labeled partition increasing from 265 ($10\%$) to 530 ($20\%$).
- **`training_started = FALSE`**: EXP-MLUA-002 has **NOT** been started and remains in **STANDBY**.
- **`real_test_evaluation_started = FALSE`**: Zero evaluation runs on the sealed 100-case test set.
- **`EXP001_interrupted = FALSE`**: EXP-MLUA-001 is actively and uninterruptedly training in the background (Task `task-2174`).
- **Final Verdict**: **`PASS`**.
