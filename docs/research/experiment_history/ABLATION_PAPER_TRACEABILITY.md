# MLUA ABLATION PAPER TRACEABILITY & MECHANISM MAPPING
**Project**: Standalone Dental Panoramic Caries Segmentation  
**Document**: `docs/ABLATION_PAPER_TRACEABILITY.md`  
**Classification**: Paper-Aligned Multi-Disturbance Framework on DC1000 Panoramic Dataset  
**Status**: VERIFIED & MAPPED

---

## 1. Traceability Architecture Overview

The Multi-Level Uncertainty Aggregation (MLUA) semi-supervised learning framework introduces three complementary disturbance mechanisms designed to perturb predictions and quantify uncertainty:
1. **Iterative Disturbance ($\mathcal{D}_{\text{iter}}$)**
2. **Noisy Disturbance ($\mathcal{D}_{\text{noise}}$)**
3. **Multi-Scale Disturbance ($\mathcal{D}_{\text{scale}}$)**

Below is the comprehensive traceability mapping connecting the theoretical formulation from the original MLUA paper to our concrete codebase implementation.

---

## 2. End-to-End Mechanism Traceability Matrix

| # | Paper Disturbance Mechanism | Paper Mathematical Formulation | Codebase Implementation | Config Flags (`configs/experiments/ablation/`) | Core Scientific Question |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **A** | **Iterative Disturbance** | $\theta_T \leftarrow \alpha \theta_T + (1-\alpha) \theta_S$<br>$\alpha = \min(1 - \frac{1}{e+1}, 0.99)$ | `p_tea.data.mul_(alpha).add_(p_stu.data, alpha=1.0-alpha)` in `train_exp001.py` | `disturbances.iterative_disturbance: bool`<br>`ssl.ema_enabled: bool`<br>`ssl.ema_theta: 0.99` | Does temporal ensembling via EMA stabilize teacher targets across training steps without mode collapse? |
| **B** | **Noisy Disturbance** | $\tilde{X}_T = \text{clamp}(X + \epsilon, 0, 1)$<br>$\epsilon \sim \mathcal{N}(0, 0.01^2)$<br>$T=8$ MC dropout passes<br>$U = -\sum \bar{P} \ln \bar{P}$<br>$E_{\text{th}} = [b(1 - \frac{e}{E}) + \frac{e}{E}]\ln(2)$ | Unified 36-image Teacher Forward under `torch.inference_mode()` + voxel entropy masking in `train_exp001.py` | `disturbances.noisy_disturbance: bool`<br>`ssl.mc_iterations: 8`<br>`ssl.noise_sigma: 0.01`<br>`ssl.dynamic_threshold: bool` | Does stochastic input noise combined with epistemic MC uncertainty filtering prevent erroneous pseudo-label reinforcement? |
| **C** | **Multi-Scale Disturbance** | Multi-level supervision across 4 pyramid scales:<br>$\mathcal{L}_{\text{seg}} = \mathcal{L}(P_{\text{fused}}) + \sum_{k=1}^4 w_k \mathcal{L}(P_{\text{aux}, k})$<br>$w = [0.1, 0.2, 0.3, 0.4]$ | `FPNDecoder` with 4 `SegmentationHead` aux heads + fused head, supervised via `bce_loss_fn` & `dice_loss_fn` | `disturbances.multiscale_disturbance: bool`<br>`loss.deep_supervision: bool`<br>`loss.aux_weights: [0.1, 0.2, 0.3, 0.4]` | Does multi-resolution feature constraint enforce structural coherence across fine and coarse caries lesions? |

---

## 3. Paper Terminology vs. Codebase Mapping

| Dimension | Paper Concept | Codebase Representation | Notes & Alignments |
| :--- | :--- | :--- | :--- |
| **Encoder Backbone** | ResNet-34 Feature Extractor | `StandaloneResNet34Encoder` in `src/mlua/models/fpn.py` | 5-stage encoder; first conv adapted for 1-channel grayscale |
| **Feature Pyramid** | Top-down pyramid decoder | `FPNDecoder` with `FPNBlock`, `SegmentationBlock` | 256 pyramid channels, 128 segmentation channels |
| **Auxiliary Predictions** | Multi-level decoder outputs | `aux_segmentation_head_list` (4 heads) | Projections from $P_2, P_3, P_4, P_5$ upsampled to $384 \times 384$ |
| **Fused Prediction** | Final merged segmentation | `segmentation_head` (`masks` tensor) | Fused multi-scale representation |
| **Uncertainty Ensemble** | Combined Multi-scale & MC passes | 5 scales $\times$ 8 MC passes = 40 prediction maps | Stacked in `torch.cat([b_final_5, b_pyr_5])` |
| **Batch Sampler** | 2-stream semi-supervised batch | `TwoStreamBatchSampler` | 4 labeled + 4 unlabeled per batch of 8 |
| **Validation Metric** | Slice / Volume Dice Evaluation | Sliding-window panoramic reconstruction ($768 \times 1536$) | 21 deterministic patches per panoramic radiograph |

---

## 4. Scientific Context & Dataset Alignment

In compliance with rigorous scientific publication standards:
1. **Modality Adaptation**: The original MLUA publication validated the framework on 3D cardiac (ACDC) and left atrium (LA) MRI/CT volumetric slices. In this project, the tri-disturbance framework is adapted for **2D Dental Panoramic Caries Segmentation** using the DC1000 benchmark dataset ($384 \times 384$ radiograph patches).
2. **Standardized Partition**: All 8 ablation runs utilize the standardized **10% SSL partition (265 labeled / 2,124 unlabeled)** at `Seed 42`.
3. **Designation**: Formally designated as a **paper-aligned ablation study**.
