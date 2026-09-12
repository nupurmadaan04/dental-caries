# MONTE CARLO SAMPLING PAPER TRACEABILITY & SENSITIVITY STUDY MAPPING
**Project**: Standalone Dental Panoramic Caries Segmentation (MLUA)  
**Document**: `docs/MC_SAMPLING_PAPER_TRACEABILITY.md`  
**Classification**: Paper-Aligned Monte Carlo Sampling Sensitivity Study on DC1000 Dataset  
**Status**: VERIFIED & DOCUMENTED

---

## 1. Scientific Overview & Theoretical Foundations

In semi-supervised medical image segmentation with teacher-student architectures, pseudo-labels derived from unlabeled data are susceptible to high epistemic and aleatoric uncertainty. The MLUA framework models epistemic uncertainty by perturbing the teacher model across multiple stochastic forward evaluations (Monte Carlo sampling) combined with Gaussian input disturbance and multi-scale decoder aggregation.

> *Statement of Alignment*: **This is a paper-aligned MC sampling sensitivity study on the current DC1000 dental panoramic implementation.**

---

## 2. End-to-End Mechanism Mapping

| Dimension | Theoretical Formulation | Source-Code Implementation | Configuration Field | Experimental Role |
| :--- | :--- | :--- | :--- | :--- |
| **MC Perturbation Passes** | $T \in \{5, 10, 20, 40, 80, 160\}$ | `t_val = cfg["mc_sampling"]["mc_samples"]` in `train_exp001.py` | `mc_sampling.mc_samples` | **Independent Experimental Variable** |
| **Input Gaussian Disturbance** | $\tilde{X}_T = \text{clamp}(X + \epsilon, 0, 1), \epsilon \sim \mathcal{N}(0, 0.01^2)$ | `torch.clamp(torch.randn(...) * 0.01, -0.1, 0.1)` | `noise.sigma: 0.01`, `noise.clamp_min/max: [-0.1, 0.1]` | Invariant across all 6 runs |
| **Teacher Model State** | Fixed weights during MC passes ($\theta_T = \text{const}$) | Evaluated inside `torch.inference_mode()` | `teacher.freeze_during_mc: true` | Invariant (No intra-batch gradient update) |
| **Multi-Scale Head Ensembling** | 5 output projections per forward pass (1 Fused + 4 Aux) | `[b_final_5, b_pyr_5]` concatenated into `(5 * T, B, 1, H, W)` | `mc_sampling.multiscale_heads_evaluated: 5` | Ensembles multi-resolution representations |
| **Voxel-Wise Entropy Uncertainty** | $U = -p \ln(p) - (1-p) \ln(1-p)$ | `-2.0 * sum(mean_preds * log(mean_preds + 1e-6))` | `uncertainty.metric: voxel_entropy` | Measures prediction variance across ensemble |
| **Dynamic Certainty Threshold** | $E_{\text{th}}(e) = [b(1 - \frac{e}{E}) + 1.0(\frac{e}{E})]\ln(2)$ | Sigmoid ramp-up from $0.520 \to 0.693$ | `uncertainty.threshold_start/end_factor: [0.75, 1.00]` | Selectively filters low-confidence voxels |
| **Consistency Target ($\sigma(P_T)$)** | Clean / Perturbed Base Teacher Target | `ul_pred_tea = all_final[:stride]` | `mc_sampling.base_evaluations: 1` | The target for Student consistency loss |

---

## 3. Exact Semantic Analysis: What Constitutes $T$?

In our source-verified MLUA implementation (`src/mlua/engine/train_exp001.py`):
1. **$T$ is the number of stochastic noisy teacher forward passes**.
2. **Total Teacher Evaluations per Unlabeled Item**: Exactly $1 + T$, comprising:
   - $1$ Base Perturbed Target Prediction ($\tilde{X}_{\text{base}} = X + \epsilon_{\text{base}}$), which serves as the actual pseudo-label target for the student consistency MSE loss: $\mathcal{L}_{\text{cons}} = \|\sigma(P_S) - \sigma(P_{T, \text{base}})\|_2^2 \cdot M$.
   - $T$ Stochastic MC Perturbation Passes ($\tilde{X}_{\text{MC}, 1..T} = X + \epsilon_{\text{MC}, t}$), which are exclusively evaluated to construct the uncertainty ensemble and compute voxel entropy $U$.
3. **Total Multi-Scale Ensemble Members**:
   $$\text{Total Ensemble Size} = 5 \times T$$
   (4 auxiliary feature heads $P_{\text{aux}, 1..4}$ + 1 fused head $P_{\text{fused}}$, evaluated across all $T$ stochastic passes).
4. **Stochasticity Sources**:
   - Additive Gaussian noise on image space ($\sigma=0.01$).
   - 2D Spatial Dropout ($p=0.2$) in the `FPNDecoder`.
   - Formally designated as: **"Monte Carlo Gaussian perturbation sampling with decoder dropout"**.

---

## 4. Expected Scientific Inquiries

This 6-arm sensitivity study investigates:
- **Accuracy vs. Sample Density**: Does increasing $T$ from 5 to 40, 80, or 160 reduce variance in entropy estimation, leading to sharper pseudo-label boundaries?
- **Diminishing Returns & Saturation**: At what threshold $T^*$ does marginal Dice gain approach zero while compute scales linearly ($\mathcal{O}(T)$)?
- **Computational Trade-Off**: Quantifying the trade-off between training duration per epoch and caries lesion detection sensitivity.
