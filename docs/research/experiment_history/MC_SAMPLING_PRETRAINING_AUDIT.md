# MONTE CARLO SAMPLING SENSITIVITY STUDY PRE-TRAINING AUDIT REPORT
**Suite**: `MC-05`, `MC-10`, `MC-20`, `MC-40`, `MC-80`, `MC-160` (6-Arm Sampling Density Matrix)  
**Document**: `docs/MC_SAMPLING_PRETRAINING_AUDIT.md`  
**Classification**: Paper-Aligned Monte Carlo Sensitivity Study on Dental Panoramic Radiographs  
**Status**: **[PASS WITH DOCUMENTED DIFFERENCES] — PREPARATION & AUDIT COMPLETED**  
**Training Execution Status**: `FALSE` (No training started; EXP-MLUA-001 continues undisturbed)

---

## 1. Executive Summary & Verification Matrix

| Verification Item | Target Specification | Actual Status | Verdict |
| :--- | :--- | :--- | :---: |
| **Config Generation** | 6 dedicated YAML files (`MC-05.yaml` .. `MC-160.yaml`) | Generated & parsed | **PASS** |
| **Unique Experiment IDs** | Exactly 6 unique IDs (`MC-05`, `MC-10`, `MC-20`, `MC-40`, `MC-80`, `MC-160`)| Verified unique | **PASS** |
| **MC Sample Counts ($T$)**| $T \in \{5, 10, 20, 40, 80, 160\}$ | Verified exact values | **PASS** |
| **Single-Variable Delta** | Only `mc_samples` varies between configs | 100% parameter invariance | **PASS** |
| **Output Directories** | `outputs/experiments/MC-XX/` (isolated subdirs) | Dedicated folders created | **PASS** |
| **Checkpoints Safety** | Zero premature checkpoints created | All folders empty | **PASS** |
| **History Schema** | `training_history.csv` with entropy/uncertainty fields | Initialized across all 6 | **PASS** |
| **Comparison Table** | `outputs/experiments/MC_SAMPLING_COMPARISON.csv` | Initialized with `NOT_RUN` | **PASS** |
| **Dynamic Batching** | Supports dynamic $(1 + T) \times B_{\text{unlabeled}}$ batching | Forward smoke test passed | **PASS** |
| **Sealed Test Safety** | `dataset/test/` (100 cases) untouched | 100% isolated & sealed | **PASS** |
| **EXP-MLUA-001 State** | Actively training in background (Task `task-2174`) | Unaltered & progressing | **PASS** |
| **EXP-MLUA-002 State** | Config and directories on Standby | Unaltered on Standby | **PASS** |
| **Ablation Suite State** | `ABL-00` .. `ABL-07` on Standby | Unaltered on Standby | **PASS** |
| **Training Execution** | Zero training iterations initiated | `training_started = FALSE`| **PASS** |

---

## 2. Configuration Files & Output Isolation

### A. Configurations Created
- [`configs/experiments/mc_sampling/MC-05.yaml`](file:///c:/Users/devin/MLUA/configs/experiments/mc_sampling/MC-05.yaml) ($T=5$ Stochastic Passes)
- [`configs/experiments/mc_sampling/MC-10.yaml`](file:///c:/Users/devin/MLUA/configs/experiments/mc_sampling/MC-10.yaml) ($T=10$ Stochastic Passes)
- [`configs/experiments/mc_sampling/MC-20.yaml`](file:///c:/Users/devin/MLUA/configs/experiments/mc_sampling/MC-20.yaml) ($T=20$ Stochastic Passes)
- [`configs/experiments/mc_sampling/MC-40.yaml`](file:///c:/Users/devin/MLUA/configs/experiments/mc_sampling/MC-40.yaml) ($T=40$ Stochastic Passes)
- [`configs/experiments/mc_sampling/MC-80.yaml`](file:///c:/Users/devin/MLUA/configs/experiments/mc_sampling/MC-80.yaml) ($T=80$ Stochastic Passes)
- [`configs/experiments/mc_sampling/MC-160.yaml`](file:///c:/Users/devin/MLUA/configs/experiments/mc_sampling/MC-160.yaml) ($T=160$ Stochastic Passes)

### B. Output Directories & Histories
Each experiment $T \in \{5, 10, 20, 40, 80, 160\}$ reserves:
- `outputs/experiments/MC-XX/checkpoints/` (Verified empty)
- `outputs/experiments/MC-XX/logs/`
- `outputs/experiments/MC-XX/evaluation/`
- `outputs/experiments/MC-XX/reports/`
- [`outputs/experiments/MC-XX/training_history.csv`](file:///c:/Users/devin/MLUA/outputs/experiments/MC-05/training_history.csv)

### C. Comparison Matrix Schema
Saved to [`outputs/experiments/MC_SAMPLING_COMPARISON.csv`](file:///c:/Users/devin/MLUA/outputs/experiments/MC_SAMPLING_COMPARISON.csv):
```csv
MC Samples,Dice,Sensitivity,Precision,IoU,F1,Specificity,Accuracy,Foreground Prevalence,Mean Uncertainty,Mean Entropy,Confidence Coverage,Training Time,Inference Time,Status
T=5,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,NOT_RUN
T=10,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,NOT_RUN
T=20,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,NOT_RUN
T=40,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,NOT_RUN
T=80,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,NOT_RUN
T=160,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,NOT_RUN
```

---

## 3. Exact MC Semantics & Implementation Mechanics

```
                         [ 4 Unlabeled Images ]
                                   │
       ┌───────────────────────────┴───────────────────────────┐
       ▼                                                       ▼
[ 1 Base Perturbed Target ]                             [ T Stochastic MC Perturbations ]
(Input: 4 images)                                       (Input: 4 * T images)
       │                                                       │
[ Teacher Forward ]                                     [ Teacher MC Forward Passes ]
       │                                                       │
       ▼                                                       ▼
[ P_T,base ] (Target for MSE Loss)                      [ 5 * T Multi-Scale Head Outputs ]
                                                               │
                                                               ▼
                                                        [ Mean Prediction (p) ]
                                                               │
                                                               ▼
                                                        [ Voxel Entropy (U) ]
                                                               │
                                                               ▼
                                                        [ Dynamic Mask (M) = I(U < E_th) ]
```

1. **Semantic Definition of $T$**: $T$ represents the number of stochastic perturbation forward passes.
2. **Total Teacher Evaluations**: $(1 + T) \times 4$ images per batch iteration.
3. **Multi-Scale Uncertainty Aggregation**: Each forward pass extracts 5 heads (1 fused + 4 aux), yielding an ensemble of $5 \times T$ maps for computing the mean prediction $\bar{p}$.
4. **Voxel Entropy Formulation**:
   $$U(x) = -\bar{p}(x) \ln(\bar{p}(x) + 10^{-6}) - (1 - \bar{p}(x)) \ln(1 - \bar{p}(x) + 10^{-6})$$
5. **Certainty Masking**:
   $$M(x) = \mathbb{I}\left(U(x) < E_{\text{th}}(e)\right), \quad E_{\text{th}}(e) = \left[ 0.75 \left(1 - \frac{e}{200}\right) + 1.0 \left(\frac{e}{200}\right) \right] \cdot \ln(2)$$
6. **Student Consistency Loss**:
   $$\mathcal{L}_{\text{cons}} = \frac{\sum_{i=1}^{B_U} M_i \cdot \|\sigma(P_{S, i}) - \sigma(P_{T, \text{base}, i})\|_2^2}{2 \sum M_i + 10^{-16}}$$

---

## 4. Controlled Invariance & Single-Variable Isolation

| Parameter | Controlled Value Across All 6 Experiments |
| :--- | :--- |
| **Dataset Pool** | DC1000 Training Patches (2,389 total) |
| **Data Partition** | Seed 42: 265 Labeled (10%) / 2,124 Unlabeled (90%) |
| **Batch Size** | 8 ($4\text{ Labeled} + 4\text{ Unlabeled}$) |
| **Backbone & Decoder**| ResNet-34 FPN with 4 Auxiliary Heads + 1 Fused Head |
| **Optimizer** | AdamW ($\text{lr}=10^{-3}, \text{wd}=10^{-2}, \beta=(0.9, 0.999)$) |
| **Scheduler** | Polynomial Decay ($p=0.9$, $\text{max\_epochs}=200$) |
| **Teacher EMA** | $\alpha = 0.99$ (Vectorized in-place update) |
| **Noise Parameters** | Gaussian $\sigma = 0.01$, clamp $[-0.1, 0.1]$ |
| **Dynamic Threshold**| Start: $0.75 \ln(2) \approx 0.520 \to$ End: $1.00 \ln(2) \approx 0.693$ |
| **Sealed Test Set** | `dataset/test/` (100 cases) untouched |

---

## 5. Compute Scaling & Memory Dynamics

| Experiment | $T$ Value | Unified Forward Images | Multi-Scale Ensemble | Est. Forward Memory Scaling |
| :--- | :---: | :---: | :---: | :---: |
| **`MC-05`** | $T=5$ | $(1+5) \times 4 = 24$ | $5 \times 5 = 25$ heads | $\sim 0.67\times$ of baseline |
| **`MC-10`** | $T=10$ | $(1+10) \times 4 = 44$ | $5 \times 10 = 50$ heads | $\sim 1.22\times$ of baseline |
| **`MC-20`** | $T=20$ | $(1+20) \times 4 = 84$ | $5 \times 20 = 100$ heads | $\sim 2.33\times$ of baseline |
| **`MC-40`** | $T=40$ | $(1+40) \times 4 = 164$ | $5 \times 40 = 200$ heads | $\sim 4.55\times$ of baseline |
| **`MC-80`** | $T=80$ | $(1+80) \times 4 = 324$ | $5 \times 80 = 400$ heads | $\sim 9.00\times$ of baseline |
| **`MC-160`**| $T=160$| $(1+160) \times 4 = 644$| $5 \times 160 = 800$ heads | $\sim 17.89\times$ of baseline |

*(Note: In GPU memory-constrained regimes for $T \ge 80$, the forward pass can optionally be executed in micro-batches of 32 images under `torch.inference_mode()` without altering the numerical result).*

---

## 6. Pre-Flight Verification Results

```
===========================================================================
EXP-MLUA MC SAMPLING SENSITIVITY STUDY PRE-TRAINING AUDIT
===========================================================================
[MC-05] Config successfully loaded: MC-05_T5
[MC-10] Config successfully loaded: MC-10_T10
[MC-20] Config successfully loaded: MC-20_T20
[MC-40] Config successfully loaded: MC-40_T40
[MC-80] Config successfully loaded: MC-80_T80
[MC-160] Config successfully loaded: MC-160_T160
[Unique IDs] Verified 6 unique experiment IDs: ['MC-05', 'MC-10', 'MC-160', 'MC-20', 'MC-40', 'MC-80']
[T Values] Verified exact MC sample counts: [5, 10, 20, 40, 80, 160]
[Invariance] Verified that all non-MC scientific parameters are 100% identical across all 6 configs.
[Directories] All 6 MC output directories & subdirectories verified clean and isolated.
[Compute Device] Testing dynamic MC scaling on cpu...
[MC-05] T=5 Passed dynamic scaling test: Input=torch.Size([24, 1, 64, 64]) -> Ensemble=torch.Size([25, 4, 1, 64, 64]) -> Mean/Uncertainty/Mask=torch.Size([4, 1, 64, 64])
[MC-10] T=10 Passed dynamic scaling test: Input=torch.Size([44, 1, 64, 64]) -> Ensemble=torch.Size([50, 4, 1, 64, 64]) -> Mean/Uncertainty/Mask=torch.Size([4, 1, 64, 64])
[MC-20] T=20 Passed dynamic scaling test: Input=torch.Size([84, 1, 64, 64]) -> Ensemble=torch.Size([100, 4, 1, 64, 64]) -> Mean/Uncertainty/Mask=torch.Size([4, 1, 64, 64])
[MC-40] T=40 Passed dynamic scaling test: Input=torch.Size([164, 1, 64, 64]) -> Ensemble=torch.Size([200, 4, 1, 64, 64]) -> Mean/Uncertainty/Mask=torch.Size([4, 1, 64, 64])
[MC-80] T=80 Passed dynamic scaling test: Input=torch.Size([324, 1, 64, 64]) -> Ensemble=torch.Size([400, 4, 1, 64, 64]) -> Mean/Uncertainty/Mask=torch.Size([4, 1, 64, 64])
[MC-160] T=160 Passed dynamic scaling test: Input=torch.Size([644, 1, 64, 64]) -> Ensemble=torch.Size([800, 4, 1, 64, 64]) -> Mean/Uncertainty/Mask=torch.Size([4, 1, 64, 64])
[Sealed Benchmark] dataset/test/ verified untouched and sealed.
[Experiment Isolation] EXP-MLUA-001, EXP-MLUA-002, and ABL-00..07 directories verified intact and untouched.

===========================================================================
ALL 6 MC SAMPLING PRE-TRAINING AUDIT CHECKS: [PASS]
===========================================================================
```

---

## 7. Final Verdict & Non-Interference Confirmation

$$\mathbf{MC\text{-}SAMPLING\text{ SENSITIVITY PREPARATION: [PASS WITH DOCUMENTED DIFFERENCES]}}$$

1. **`training_started = FALSE`**: Zero training runs have been executed for `MC-05` through `MC-160`.
2. **EXP-MLUA-001 Intact**: `EXP-MLUA-001` is actively training undisturbed in the background (Task `task-2174`).
3. **EXP-MLUA-002 & ABL-00..07 Intact**: All prior prepared configurations and directory trees remain completely unaltered on **STANDBY**.
4. **Sealed Benchmark Intact**: `dataset/test/` (100 cases) remains 100% sealed and untouched.
