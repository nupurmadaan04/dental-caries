# DAY 2 — TEACHER MC COMPUTATION PROFILING TECHNICAL NOTE

---

## 1. Executive Summary & Objective

This profiling note provides an in-depth computational and structural breakdown of the Teacher Monte Carlo (MC) inference loop in **EXP-MLUA-001**. 

In the Day 1 forensic benchmark, Teacher MC sampling ($T=8$) was identified as the primary computational bottleneck, consuming **61.5% of total batch runtime (14.54s out of 23.61s baseline)**. The initial batched-MC pass reduced this to **11.54s per batch (1.26× speedup)**.

The objective of Day 2 is to investigate whether additional safe, mathematically equivalent computational optimizations can further accelerate the Teacher MC pass without altering any scientific parameters.

---

## 2. Current Execution Flow & Architecture

Per training batch ($B=8$, composed of 4 labeled + 4 unlabeled patches of size $1 \times 384 \times 384$):

```
+---------------------------------------------------------------------------------------------------+
|                                CURRENT TRAINING BATCH EXECUTION FLOW                              |
+---------------------------------------------------------------------------------------------------+
| 1. Data Loader Batch: [8, 1, 384, 384] (4 Labeled, 4 Unlabeled)                                   |
|                                                                                                   |
| 2. Student Forward Pass (train mode, autograd on):                                                |
|    - Input: [8, 1, 384, 384]                                                                      |
|    - Outputs: pred_fused [8, 1, 384, 384], pred_aux_list (4 heads each [8, 1, 384, 384])        |
|                                                                                                   |
| 3. Teacher Forward Pass #1 (Base Consistency Target):                                             |
|    - Input: volume_batch_r = ul_data + noise_base -> [4, 1, 384, 384]                             |
|    - Model call: model_tea(volume_batch_r) under torch.no_grad()                                  |
|    - Output: ul_pred_tea [4, 1, 384, 384] (consistency target)                                    |
|                                                                                                   |
| 4. Teacher Forward Pass #2 (Batched MC Passes, T=8):                                              |
|    - Input: ema_inputs_all = (ul_data.unsqueeze(0) + noises).view(32, 1, 384, 384)                |
|    - Model call: model_tea(ema_inputs_all) under torch.no_grad()                                  |
|    - Outputs: final_pred_all [32, 1, 384, 384], pyramid_pred_all_list (4 heads each [32, ...])   |
|                                                                                                   |
| 5. Multi-Level Uncertainty Estimation:                                                            |
|    - Reshape 5 output streams (1 fused + 4 pyramid levels) over T=8 passes -> [40, 4, 1, 384, 384]|
|    - Sigmoid & Mean: mean_preds = torch.mean(all_preds_5, dim=0).sigmoid() -> [4, 1, 384, 384]   |
|    - Information Entropy: -2.0 * sum(mean_preds * log(mean_preds + 1e-6)) -> [4, 1, 384, 384]   |
|    - Thresholding: mask = (uncertainty < dynamic_threshold).float()                               |
|                                                                                                   |
| 6. Supervised + Consistency Loss & Backward:                                                      |
|    - 0.5 * (BCE/4 + Dice/4) + lambda * consistency_loss                                           |
|    - total_loss.backward() -> optimizer.step()                                                    |
|                                                                                                   |
| 7. EMA Update:                                                                                    |
|    - In-place parameter blend: p_tea.mul_(alpha).add_(p_stu, alpha=1-alpha)                       |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Teacher Evaluations & Tensor Shapes Breakdown

| Step | Operation | Input Shape | Output Shapes | Evaluated Images |
|---|---|---|---|---|
| **Base Target** | `model_tea(ul_data + noise_base)` | `[4, 1, 384, 384]` | Fused: `[4, 1, 384, 384]`<br>Aux: $4 \times [4, 1, 384, 384]$ | **4 images** |
| **MC Passes ($T=8$)** | `model_tea(ema_inputs_all)` | `[32, 1, 384, 384]` | Fused: `[32, 1, 384, 384]`<br>Aux: $4 \times [32, 1, 384, 384]$ | **32 images** |
| **Combined** | Current Teacher Pipeline | — | 5 heads $\times$ 36 images | **36 images / batch** |

---

## 4. Current Bottleneck Timing & Profile

Measured on CPU (8 OpenMP/MKL worker threads, 66 batches/epoch):

- **Total Baseline Batch Time:** `23.61s` (~25.97 min/epoch)
- **Teacher MC Time (Sequential $T=8$):** `14.54s` (61.5% of total compute)
- **Teacher MC Time (Batched $T=8$):** `11.54s` (55.4% of total compute)
- **Remaining Teacher Base Pass Overhead:** Separate forward dispatch for 4 images (~1.2s – 1.4s)
- **Tensor Allocation / Reshape / Entropy Math Overhead:** ~0.25s per batch

---

## 5. Candidate Optimization Opportunities

We identify 4 safe computational optimization candidates:

### Candidate A: Unified Single-Pass Teacher Inference (36 Images)
- **Concept:** Concatenate the base consistency input `volume_batch_r` ($4$ images) and the $8$ MC perturbation inputs `ema_inputs_all` ($32$ images) into a single tensor of shape `[36, 1, 384, 384]`.
- **Engineering Advantage:** Merges 2 separate forward dispatches into 1 single forward pass. Reduces kernel launch overhead and allows OpenMP/MKL GEMM threads to maximize cache reuse.
- **Numerical Risk:** **Zero / Negligible** ($< 10^{-7}$ floating point precision).

### Candidate B: `torch.inference_mode(True)`
- **Concept:** Replace `with torch.no_grad():` with `with torch.inference_mode():`.
- **Engineering Advantage:** In PyTorch, `inference_mode` disables autograd graph tracking AND completely bypasses C++ version counter increments and view metadata creation.
- **Numerical Risk:** **Zero** (Exact mathematical identity).

### Candidate C: Contiguous Pre-allocated Tensor Buffers
- **Concept:** Eliminate redundant intermediate tensor memory allocations during `(ul_data.unsqueeze(0) + noises).view(...)` by writing directly into a pre-allocated contiguous input buffer.
- **Engineering Advantage:** Eliminates dynamic memory allocation churn on CPU.
- **Numerical Risk:** **Zero**.

### Candidate D: Fused Entropy and Thresholding
- **Concept:** Vectorize the Shannon entropy formulation `H(p) = -2.0 * sum(p * log(p + eps))` using optimized arithmetic to avoid temporary tensor copies.
- **Engineering Advantage:** Minor CPU cache locality gain.
- **Numerical Risk:** **Zero**.

---

## 6. Expected Numerical Risk & Safety Assessment

| Optimization Candidate | Numerical Risk Level | Mathematical Equivalence | Safety Action |
|---|---|---|---|
| **Unified 36-image Pass** | Negligible ($\le 10^{-7}$) | Exact | Verify against reference batched MC |
| **`torch.inference_mode`** | None (0.00) | Exact | Verify identical outputs |
| **Contiguous Input Buffer** | None (0.00) | Exact | Verify identical outputs |
| **Fused Entropy Math** | None (0.00) | Exact | Verify mask mismatch == 0 |

---

## 7. Next Step

Construct a dedicated Day 2 validation and benchmark script comparing the Reference Batched-MC against the Candidate Optimizations on identical inputs and seeds, and publish the results to `docs/DAY2_MC_OPTIMIZATION_REPORT.md`.
