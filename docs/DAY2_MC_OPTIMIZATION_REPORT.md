# Day 2 Teacher MC Optimization Report

---

## 1. Objective

The objective of Day 2 Step 1 is to investigate whether the **Teacher Monte Carlo (MC) inference loop** in **EXP-MLUA-001** can be accelerated beyond the initial batched-MC baseline **strictly through compute-level optimizations without altering any scientific parameters, loss functions, architecture, or data splits**.

---

## 2. Original Implementation

In the baseline and initial batched-MC implementation:
1. **Teacher Consistency Base Pass:** Evaluates 4 unlabeled images `ul_data + noise_base` via `model_tea(volume_batch_r)` under `torch.no_grad()` to compute the consistency target `ul_pred_tea`.
2. **Teacher MC Uncertainty Pass:** Evaluates 32 perturbed images `(ul_data.unsqueeze(0) + noises).view(32, 1, 384, 384)` in a second separate forward call to `model_tea()`.
3. **Reshape & Entropy:** Reshapes the 5 output streams across 8 iterations into `[40, 4, 1, 384, 384]`, computes sigmoid mean, and calculates Shannon entropy with dynamic thresholding.

This structure performed **2 separate forward dispatches per batch**, creating redundant tensor allocations and PyTorch operator overhead.

---

## 3. Bottleneck Analysis

Under CPU execution, executing two distinct calls to `model_tea()` incurs:
- Redundant C++ operator launch overhead and memory tracking.
- Interrupted OpenMP/MKL GEMM thread scheduling across convolutions.
- Dynamic temporary tensor allocations (`.unsqueeze(0)` and intermediate cat/view operations).
- Standard `torch.no_grad()` metadata tracking overhead.

---

## 4. Optimization Attempt

We implemented and validated **Unified 36-Image Teacher Inference with `torch.inference_mode`**:
1. **Unified Forward Pass:** Pre-allocates a single contiguous buffer `[36, 1, 384, 384]` (`4` base + `32` MC images) and executes **1 single forward pass** through the teacher network.
2. **`torch.inference_mode(True)`:** Replaced `torch.no_grad()` to disable autograd version counter updates and tensor view metadata creation entirely.
3. **Direct Slice Extraction:** Slices `all_final[:4]` for the consistency target and `all_final[4:]` / `[h[4:] for h in all_pyramid]` for the MC uncertainty calculations with zero memory copying.
4. **RAM Pre-Caching:** Retained uint8 RAM cache of the 2,389 training patches to eliminate disk I/O latency.

---

## 5. Numerical Equivalence

Direct comparison between the **Reference Batched-MC** and the **Optimized Unified 36-Image MC** on identical input batches, random seeds, and Gaussian perturbations:

| Metric | Reference | Optimized | Difference | Pass/Fail |
|---|---|---|---|---|
| **Mean Probability** | `0.50936532` | `0.50936532` | `0.00e+00` | **PASS** |
| **Max Probability Difference** | — | — | `0.00e+00` | **PASS** |
| **Mean Uncertainty** | `0.68642676` | `0.68642676` | `0.00e+00` | **PASS** |
| **Max Uncertainty Difference** | — | — | `0.00e+00` | **PASS** |
| **Max Consistency Target Diff** | — | — | `0.00e+00` | **PASS** |
| **Uncertainty Mask Mismatch** | `0 pixels` | `0 pixels` | **0 pixels** | **PASS** |
| **Consistency Loss Difference** | — | — | `0.00e+00` | **PASS** |

> **Verification:** The optimized implementation is **bitwise identical (0.00e+00 difference)** across all probabilities, uncertainty metrics, binary masks, and consistency loss values.

---

## 6. Runtime Benchmark

Measured across **12 deterministic batches** on CPU (8 OpenMP/MKL worker threads, 66 batches/epoch):

| Implementation | sec/batch | Teacher MC (s) | Epoch Estimate (min) | 25 Epochs (hr) | 30 Epochs (hr) | 40 Epochs (hr) | 200 Epochs (hr) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline (Original Day 1)** | 23.61s | 14.54s | 25.97 min | 10.82 hr | 12.98 hr | 17.31 hr | 86.56 hr |
| **Reference Batched-MC (2 Calls)** | 21.95s | 12.18s | 24.14 min | 10.06 hr | 12.07 hr | 16.09 hr | 80.47 hr |
| **Optimized Unified 36-Image MC** | **16.71s** | **9.26s** | **18.39 min** | **7.66 hr** | **9.19 hr** | **12.26 hr** | **61.28 hr** |

### **Summary of Gains:**
- **Teacher MC Execution:** Reduced from `12.18s` to `9.26s` per batch (**1.32× speedup / 24.0% reduction**).
- **Total Batch Time:** Reduced from `21.95s` to `16.71s` per batch (**1.31× speedup / 23.8% reduction**).
- **Epoch Wall-Clock Time:** Reduced from `24.14 min` to **`18.39 min`** (saving **~5.75 minutes per epoch**).
- **Full 200-Epoch Training Runtime:** Reduced from `80.47 hours` to **`61.28 hours`** (saving **~19.2 hours**).

---

## 7. Memory Usage

| Component | RAM Footprint | Safety & Stability |
|---|---|---|
| **Dataset RAM Cache (2,389 Patches + Masks)** | ~704 MB | Stable, zero memory leaks |
| **Unified `[36, 1, 384, 384]` Input Tensor** | ~21.2 MB | Minimal CPU memory footprint |
| **Peak Process RAM During Training** | ~2.15 GB | Well within host available RAM |

---

## 8. Scientific Integrity

The scientific pipeline is **100% FROZEN AND UNTOUCHED**:
- **Architecture:** ResNet-34 FPN with 4 auxiliary heads + 1 fused head (unchanged).
- **Dataset:** DC1000 patches (2,389 total; 265 labeled [10%], 2,124 unlabeled [90%]) (unchanged).
- **Batch Size:** 8 (4 labeled + 4 unlabeled) (unchanged).
- **Monte Carlo Uncertainty:** $T=8$, Gaussian perturbation $\sigma=0.01$, clamp $[-0.1, 0.1]$ (unchanged).
- **Dynamic Thresholding:** Entropy threshold schedule $0.520 \to 0.693$ (unchanged).
- **Supervised & Consistency Loss:** Deep supervision (0.5 BCE + 0.5 Dice) + Sigmoid MSE (unchanged).
- **Optimizer & Scheduler:** AdamW (lr=0.001, weight_decay=0.01), Polynomial LR decay (unchanged).
- **EMA:** $\alpha=0.99$ (unchanged).
- **Validation Protocol:** 50 held-out training patches, decision threshold 0.5 (unchanged).
- **Sealed Test Set (`dataset/test/`):** **100% UNTOUCHED / UNACCESSED**.

---

## 9. Decision

**PASS** — The Unified 36-Image Teacher Forward Pass with `torch.inference_mode` is **bitwise identical (0.00e+00 error)** and yields a **23.8% faster total batch time (16.71s vs 21.95s)**, reducing per-epoch runtime to **18.39 minutes**.

---

## 10. Recommended Training Configuration

The Unified 36-image Teacher Forward optimization should be used for all future MLUA training runs:
- Pre-allocated `[36, 1, 384, 384]` tensor under `torch.inference_mode()`.
- RAM pre-cached dataset.
- In-place vectorized EMA parameter update.
