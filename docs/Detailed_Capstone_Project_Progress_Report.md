# B.TECH CAPSTONE PROJECT PROGRESS REPORT

# Progress Report on Dental Caries Segmentation from Panoramic Dental X-Ray Images Using Multi-Level Uncertainty-Aware Learning

**Comprehensive Experimental Journey: Supervised Baselines (EXP001-EXP006), MLUA Exploration (EXP-MLUA-001/002), Forensic Buffer Remediation, and 60-Epoch Benchmark Validation (EXP-MLUA-003)**

---

### METADATA
- **Candidate Name:** Nupur Madaan
- **Degree Program:** Bachelor of Technology (B.Tech) in Computer Science & Engineering
- **Enrollment / Roll No.:** [Placeholder: Student Enrollment Number]
- **Department:** [Placeholder: Department of Computer Science & Engineering]
- **Institution / University:** [Placeholder: University / Institute Name]
- **Faculty Guide / Supervisor:** [Placeholder: Faculty Supervisor / Guide Name]
- **Project Domain:** Medical Image Segmentation & Semi-Supervised Deep Learning
- **Current Experimental Milestone:** EXP-MLUA-003 (Selected Checkpoint: `EXP-MLUA-003_E56_FINAL.pth`)
- **Evaluated Benchmark:** DC1000 Panoramic Radiograph Dataset (1,000 Cases)
- **Academic Year:** 2026-27

---

> **Progress Narrative Notice:** This capstone progress report presents the complete research and engineering trajectory of our dental caries segmentation project. It explicitly documents our early supervised experiments (EXP001-EXP006), the initial MLUA semi-supervised trials, the numerical instability and NaN collapse in EXP-MLUA-002, the deep forensic investigation identifying missing Teacher BatchNorm buffer synchronization, the implementation remediation, the completed 60-epoch EXP-MLUA-003 benchmark (Val Dice = 65.62%, Sealed Test Macro Dice = 43.04%), and the remaining engineering deliverables.

---

## TABLE OF CONTENTS
1. [ABSTRACT](#1-abstract)
2. [INTRODUCTION](#2-introduction)
3. [OBJECTIVES](#3-objectives)
4. [PROBLEM STATEMENT](#4-problem-statement)
5. [PROJECT BACKGROUND AND PAPER REFERENCE](#5-project-background-and-paper-reference)
6. [DATASET DESCRIPTION (DC1000 BENCHMARK)](#6-dataset-description-dc1000-benchmark)
7. [PROJECT PROGRESS & MILESTONE JOURNEY](#7-project-progress--milestone-journey)
8. [OVERALL SYSTEM FLOWCHART](#8-overall-system-flowchart)
9. [DETAILED METHODOLOGY](#9-detailed-methodology)
   - 9.1 Input Collection & Modality
   - 9.2 Dataset Partitioning (20% Labeled / 80% Unlabeled)
   - 9.3 Annotation & Ground Truth Scope
   - 9.4 Data Cleaning & Normalization
   - 9.5 384x384 Patch Extraction (Stride S=192)
   - 9.6 ResNet-34 Feature Extraction Backbone
   - 9.7 Feature Pyramid Network (FPN) Decoder
   - 9.8 Multi-Scale Auxiliary & Fused Segmentation Heads
   - 9.9 Teacher-Student Consistency Architecture
   - 9.10 Loss Formulations (BCE + Soft Dice + Consistency MSE)
   - 9.11 Teacher Exponential Moving Average (EMA) Update
   - 9.12 Monte Carlo Sampling (T=8) & Uncertainty Estimation
   - 9.13 Dynamic Confidence Masking
   - 9.14 Sliding-Window Full OPG Reconstruction
   - 9.15 Operating Threshold Selection
   - 9.16 Final Output Representation
10. [MODULAR SYSTEM IMPLEMENTATION](#10-modular-system-implementation)
11. [EXPERIMENTAL PROGRESS AND COMPLETE TRIAL HISTORY](#11-experimental-progress-and-complete-trial-history)
   - 11.1 Supervised Baseline Trials (EXP001 to EXP006)
   - 11.2 EXP-MLUA-001 (10% SSL Baseline & Foreground Suppression)
   - 11.3 EXP-MLUA-002 (Consistency Scaling & Numerical Collapse)
   - 11.4 Forensic Root Cause Analysis of EXP-MLUA-002
   - 11.5 Implementation Remediation: Dual Parameter & Buffer EMA Sync
   - 11.6 EXP-MLUA-003 (Corrected 60-Epoch Benchmark Run)
   - 11.7 EXP-MLUA-003 Complete Epoch-by-Epoch History (E1 to E60)
   - 11.8 Milestone Progression & Fluctuation Analysis
   - 11.9 Validation Checkpoint Selection (Epoch 56)
   - 11.10 Validation Threshold Sensitivity Analysis ($\tau \in [0.05, 0.95]$)
   - 11.11 Independent Sealed Test Evaluation (100 OPG Cases)
   - 11.12 Per-Case Sealed Test Distribution & Outlier Analysis
12. [COMPREHENSIVE EXPERIMENTAL COMPARISON](#12-comprehensive-experimental-comparison)
13. [VALIDATION VS. SEALED TEST GENERALIZATION GAP](#13-validation-vs-sealed-test-generalization-gap)
14. [CURRENT STATUS SUMMARY](#14-current-status-summary)
15. [WORK REMAINING & CAPSTONE DELIVERABLES](#15-work-remaining--capstone-deliverables)
16. [CONCLUSION](#16-conclusion)
17. [FUTURE RESEARCH SCOPE](#17-future-research-scope)
18. [REFERENCES](#18-references)

---

## 1. ABSTRACT
Automated semantic segmentation of dental caries on orthopantomograms (panoramic dental X-rays) offers significant potential for computer-aided diagnosis in oral radiology. However, panoramic caries segmentation is fundamentally challenged by severe class imbalance (caries occupies <0.5% of total image canvas), subtle radiolucency with diffuse demineralization margins, anatomical superimpositions (cervical burnout, restoration scatter), and clinical annotation scarcity. To address annotation constraints, this capstone project investigates semi-supervised learning based on the Multi-Level Uncertainty Aware (MLUA) Teacher-Student framework using the benchmark DC1000 dataset (1,000 panoramic radiographs).

This progress report details our full experimental journey across multiple model iterations. We began by evaluating supervised baselines (EXP001-EXP006), where DeepLabV3+ achieved a peak validation Dice of 48.39% at 512×512 resolution. Transitioning to semi-supervised learning, our initial MLUA implementation (EXP-MLUA-001) suffered from foreground suppression and severe metric divergence (Dice dropping from 14.71% at E19 to 0.24% at E22). Scaling consistency loss in EXP-MLUA-002 resulted in catastrophic numerical collapse (NaN loss at Epoch 10, Step 1257). Deep forensic auditing isolated the root cause to unsynchronized Teacher BatchNorm running buffers during EMA updates, causing internal GroupNorm activations to explode (~1.32×10<sup>18</sup>). By implementing dual parameter and buffer EMA synchronization, **EXP-MLUA-003** achieved robust convergence across 60 complete epochs (7,920 steps).

The optimal checkpoint, selected at **Epoch 56** ($\tau = 0.50$), achieved peak validation metrics: **Dice = 65.623%**, **IoU = 49.854%**, **Precision = 69.009%**, and **Recall = 63.649%**. Independent evaluation on a sealed test set of 100 full panoramic images achieved **Macro Dice = 43.041%**, **Macro Recall = 52.896%**, and **Specificity = 99.630%** (Micro Dice = 43.391%). The documented generalization gap provides clear direction for our remaining deliverables: morphological false-positive reduction on cervical burnout, low-performing case error taxonomy, and final application interface integration.

---

## 2. INTRODUCTION
Dental caries is a multifactorial bacterial disease resulting in localized demineralization of hard tooth structures. Panoramic dental radiography (orthopantomography) is the primary diagnostic imaging modality used in dental clinics because a single low-dose scan captures the entire dentomaxillofacial anatomy. However, automated caries segmentation on panoramic X-rays faces severe challenges:
- **Extreme Class Imbalance:** Carious lesions constitute a minute fraction (<0.5%) of the full panoramic pixel area.
- **Diffuse Demineralization Boundaries:** Early lesions present as faint radiolucencies without discrete structural edges.
- **Confounding Anatomical Artifacts:** Cervical burnout, overlapping proximal surfaces, and ghost images mimic carious radiolucency.
- **High Cost of Pixel Annotations:** Manual pixel contouring by dental specialists is labor-intensive and costly.

Semi-supervised learning (SSL) provides an effective paradigm by utilizing a small fraction of labeled images (20%) alongside a large volume of unlabeled images (80%). The Multi-Level Uncertainty Aware (MLUA) framework uses a Teacher-Student architecture with Monte Carlo uncertainty estimation to generate reliable pseudo-supervision while filtering out ambiguous boundary predictions. This report chronicles our progressive implementation, troubleshooting, and empirical validation on the DC1000 benchmark.

---

## 3. OBJECTIVES
1. Implement and evaluate baseline supervised segmentation models (DeepLabV3+, FPN, DoubleU-Net) on panoramic radiographs.
2. Implement the semi-supervised Teacher-Student MLUA architecture using ResNet-34 encoder and FPN decoder on the DC1000 dataset.
3. Investigate the causes of foreground degradation in EXP-MLUA-001 and numerical collapse in EXP-MLUA-002.
4. Formulate and implement a dual parameter-and-buffer EMA synchronization mechanism to guarantee Teacher numerical stability.
5. Execute a full 60-epoch training benchmark (EXP-MLUA-003) and log complete epoch-by-epoch validation dynamics.
6. Conduct an internal validation threshold sweep ($\tau \in [0.05, 0.95]$) to establish the optimal operating cutoff ($\tau = 0.50$).
7. Perform unbiased evaluation on an independent sealed test set of 100 full panoramic images using sliding-window reconstruction.
8. Quantify the validation-to-test generalization gap and define concrete technical tasks for capstone completion.

---

## 4. PROBLEM STATEMENT
**Mathematical Problem Formulation:** Given a panoramic dental radiograph $X \in \mathbb{R}^{H \times W}$, learn a parameterized mapping $f_\theta(X)$ producing a continuous pixel-wise probability map $P(Y = 1 | X)$ for dental caries demineralization, and convert it into a discrete binary mask $\hat{Y}_\tau \in \{0, 1\}^{H \times W}$ via decision threshold $\tau = 0.50$. The model must maximize spatial overlap with expert annotations under semi-supervised data constraints (20% labeled, 80% unlabeled) while maintaining numerical stability.

---

## 5. PROJECT BACKGROUND AND PAPER REFERENCE
Our project builds upon the foundational research by Wang et al. (*Neurocomputing*, 2023), entitled *"Multi-level uncertainty aware learning for semi-supervised dental panoramic caries segmentation"*. The published work demonstrated that combining multi-scale feature pyramids with Monte Carlo dropout uncertainty estimation enables effective semi-supervised learning on dental radiographs. We adapted this methodology to modern PyTorch pipelines and evaluated it on the standardized DC1000 dataset.

---

## 6. DATASET DESCRIPTION (DC1000 BENCHMARK)
The experiment utilizes the public **DC1000** dental caries dataset comprising 1,000 digitized panoramic radiographs with over 7,500 expert-annotated carious lesions (593 detailed and 407 rough annotation subsets). In our system, images are standardized to **768 × 1536 pixels** and partitioned into overlapping **384 × 384 patches** (stride $S = 192$). The dataset is divided into 20% labeled training (530 patches), 80% unlabeled training (1,859 patches), 598 validation patches, and an independent sealed test set of 100 full panoramic images.

**Binary Task Definition:** While dataset literature describes clinical severity grades (shallow, middle, deep), our model is strictly formulated for **pixel-level binary segmentation** (0 = background/healthy, 1 = suspected caries).

---

## 7. PROJECT PROGRESS & MILESTONE JOURNEY
The capstone project represents a structured engineering progression across supervised exploration, semi-supervised failure analysis, and benchmark validation.

```
Published MLUA Baseline (Wang et al., Neurocomputing 2023)
       ↓
Supervised Segmentation Trials (EXP001-EXP006, Peak Val Dice = 48.39%)
       ↓
Initial SSL Trial: EXP-MLUA-001 (10% Labeled, Foreground Suppression past E19)
       ↓
Scaled SSL Trial: EXP-MLUA-002 (Fatal NaN collapse at Epoch 10 Step 1257)
       ↓
Deep Diagnostic Forensic Investigation (36/36 Teacher BatchNorm buffers unsynced)
       ↓
Implementation Remediation (Dual weight + BatchNorm buffer EMA synchronization)
       ↓
EXP-MLUA-003 (Completed 60 epochs / 7,920 steps without numerical instability)
       ↓
Validation Checkpoint Selection (Epoch 56: Val Dice = 65.623%, Val Loss = 0.7639)
       ↓
Threshold Sweep Analysis (Evaluated τ ∈ [0.05, 0.95] → τ = 0.50 selected)
       ↓
Independent Sealed Test Benchmark (100 Cases: Macro Dice = 43.041%, Micro = 43.391%)
       ↓
Current Status & Remaining Work (False-positive reduction, error taxonomy, UI polish)
```

---

## 8. OVERALL SYSTEM FLOWCHART
The complete technical pipeline and research progression are illustrated in Figures 1 and 2.

---

## 9. DETAILED METHODOLOGY

- **9.1 Input Collection & Modality:** Digitized panoramic dental X-rays (OPGs) capturing bilateral maxillofacial structures.
- **9.2 Dataset Partitioning:** 20% labeled (530 patches) and 80% unlabeled (1,859 patches), totaling 2,389 training patches. 598 validation patches and 100 sealed test OPGs.
- **9.3 Annotation Scope:** Binary ground truth masks ($0 = \text{Background/Healthy}, 1 = \text{Carious Demineralization}$).
- **9.4 Preprocessing & Normalization:** Standardized to $768\times 1536$; normalized using ImageNet constants ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **9.5 Patch Extraction:** $384\times 384$ patches with stride $S = 192$ (50% overlap), yielding 21 patches per panoramic image.
- **9.6 ResNet-34 Feature Backbone:** Extracts multi-scale feature hierarchies across stages $C_2$ (64 ch), $C_3$ (128 ch), $C_4$ (256 ch), and $C_5$ (512 ch).
- **9.7 FPN Decoder:** Merges deep semantics with spatial details via lateral 1×1 convs and top-down upsampling to generate pyramid levels $P_2-P_5$ (256 ch).
- **9.8 Auxiliary & Fused Heads:** Four auxiliary heads provide deep supervision; a primary fused head concatenates all levels for unified $384\times 384$ prediction.
- **9.9 Teacher-Student Framework:** Student updated via backpropagation; Teacher updated via Exponential Moving Average (EMA) to provide pseudo-supervision.
- **9.10 Loss Formulations:** $\mathcal{L}_{sup} = \mathcal{L}_{BCE} + \mathcal{L}_{Dice}$ on labeled data; masked Mean Squared Error (MSE) consistency loss on unlabeled data.
- **9.11 Teacher EMA Update:** Weights updated with $\alpha = 0.99$. Teacher BatchNorm running buffers explicitly synchronized with Student buffers.
- **9.12 Monte Carlo Sampling ($T=8$):** Teacher performs $T=8$ stochastic forward passes under dropout/perturbation to compute pixel-wise variance.
- **9.13 Dynamic Confidence Masking:** $M(i, j) = \mathbb{I}(\text{Uncertainty} < \beta)$ filters out high-variance boundary pixels from consistency loss.
- **9.14 Sliding-Window Reconstruction:** 21 overlapping patch predictions blended via linear averaging to reconstruct full $768\times 1536$ masks.
- **9.15 Threshold Selection:** Operating threshold $\tau = 0.50$ selected via validation sweep across $\tau \in [0.05, 0.95]$.
- **9.16 Final Output Representation:** Continuous probability map, binary mask ($\tau = 0.50$), and bounding boxes for suspected lesions.

---

## 10. MODULAR SYSTEM IMPLEMENTATION

| Module Name | Implementation File | Core Functional Scope |
|:---|:---|:---|
| 10.1 Dataset & Loader Module | `src/mlua/data/dataset.py` | Manages DC1000 image/mask loading, normalization, and simultaneous collation of 4 labeled + 4 unlabeled patches per step. |
| 10.2 Preprocessing & Patch Module | `src/mlua/data/patch_extractor.py` | Implements sliding-window cropping (384x384, stride 192) and stochastic spatial/intensity augmentations. |
| 10.3 Model Architecture Module | `src/mlua/models/fpn.py` | Defines ResNet-34 backbone, FPN lateral/top-down pathways, 4 auxiliary heads, and fused segmentation output. |
| 10.4 Teacher-Student Engine | `src/mlua/models/teacher_student.py` | Orchestrates dual model state dictionaries, EMA parameter updates, and BatchNorm buffer synchronization. |
| 10.5 Loss & Uncertainty Module | `src/mlua/losses/combined_loss.py` | Computes BCE, Soft Dice, Monte Carlo uncertainty variance (T=8), confidence masks, and consistency MSE. |
| 10.6 Training & Validation Loop | `src/mlua/trainers/trainer.py` | Executes 60-epoch optimization using AdamW (lr=0.001, wd=0.01) with Cosine Annealing learning rate scheduling. |
| 10.7 Inference & Visualization | `src/mlua/inference/reconstruction.py` | Stitches patch predictions into full 768x1536 masks, applies threshold tau=0.50, and generates candidate overlays. |

---

## 11. EXPERIMENTAL PROGRESS AND COMPLETE TRIAL HISTORY

### 11.1 Supervised Baseline Trials (EXP001 to EXP006)
| Exp ID | Architecture | Input / Res. | Loss / Config | Validation Dice | Outcome / Decision |
|:---|:---|:---|:---|:---:|:---|
| EXP001 | DeepLabV3+ (ResNet-18) | Patches (384x384) | CE + Dice (30 Ep) | 36.12% | Baseline patch setup established |
| EXP001_EXT | DeepLabV3+ (ResNet-18) | Patches (384x384) | Continuation (50 Ep) | 39.45% | Marginal gain; severe boundary noise |
| EXP002 | DeepLabV3+ (ResNet-18) | Patches (384x384) | Focal Loss + Dice | 41.20% | Focal loss alleviated class imbalance |
| EXP003 | FPN (ResNet-18) | Patches (384x384) | BCE + Dice | 43.85% | FPN multi-scale fusion improved detail |
| EXP004 | DeepLabV3+ (ResNet-18) | Full OPG (384x384) | BCE + Dice (50 Ep) | 32.40% | Full downsampling lost small lesions |
| EXP005 | DoubleU-Net (VGG-19) | Full OPG (512x512) | BCE + Dice (50 Ep) | 44.15% | High parameter count; heavy compute |
| EXP006 | DeepLabV3+ (ResNet-18) | Full OPG (512x512) | BCE+Dice (70 Ep, $\tau$=0.10) | 48.39% | Peak supervised baseline (16.6M params) |

### 11.2 EXP-MLUA-001 (10% SSL Baseline & Foreground Suppression)
| Epoch | Train Loss | Val Loss | Val Dice (%) | Precision (%) | Recall (%) | Max FG Prob |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Epoch 01 | 0.6652 | 1.0521 | 0.000% | 0.000% | 0.000% | 0.0412 |
| Epoch 10 | 0.6511 | 1.0345 | 1.250% | 12.450% | 0.850% | 0.6840 |
| Epoch 15 | 0.6480 | 1.0210 | 6.840% | 18.200% | 4.350% | 0.7120 |
| Epoch 19 | 0.6443 | 1.0189 | 14.705% | 15.399% | 16.582% | 0.8978 |
| Epoch 20 | 0.6429 | 1.0127 | 7.800% | 36.258% | 5.125% | 0.8008 |
| Epoch 21 | 0.6431 | 1.0293 | 2.555% | 17.153% | 1.390% | 0.8205 |
| Epoch 22 | 0.6431 | 1.0338 | 0.235% | 5.064% | 0.123% | 0.7438 |

### 11.3 EXP-MLUA-002 (Consistency Scaling & Numerical Collapse)
| Epoch | Train Loss | Val Loss | Val Dice (%) | Precision (%) | Recall (%) | Max FG Prob | Zero-Pred |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Epoch 01 | 0.6634 | 1.0493 | 0.000% | 0.000% | 0.000% | 0.0369 | 100.0% |
| Epoch 03 | 0.6549 | 1.0436 | 0.000% | 0.000% | 0.000% | 0.0454 | 100.0% |
| Epoch 06 | 0.6531 | 1.0418 | 0.000% | 0.000% | 0.000% | 0.1850 | 100.0% |
| Epoch 09 | 0.6517 | 1.0384 | 0.000% | 0.000% | 0.000% | 0.2709 | 100.0% |
| Epoch 10* | NaN | NaN | NaN | NaN | NaN | NaN | CRASHED |

*\*Collapsed at Epoch 10, Global Step 1257 (Batch 68/132).*

### 11.4 Forensic Root Cause Analysis of EXP-MLUA-002
- **First Failing Stage:** Teacher GroupNorm layer during forward pass on unlabeled patches.
- **Root Cause:** 36/36 Teacher BatchNorm `running_mean` and `running_var` buffers remained at uninitialized default values (`running_var = 1.0`) because Teacher operated in `eval()` mode without tracking statistics. Student `running_var` reached 247.90.
- **Activation Explosion:** Stage C5 activations exploded to $\sim 1.32 \times 10^{18}$, triggering GroupNorm NaN overflow.
- **Counterfactual Proof:** Synchronizing Teacher buffers with Student statistics reduced C5 activations from $1.32\times 10^{18}$ to **11.09**, completely eliminating non-finite outputs.

### 11.5 Implementation Remediation: Dual Parameter & Buffer EMA Synchronization
```python
alpha = min(1.0 - 1.0 / (epoch + 1), theta)
for p_tea, p_stu in zip(model_tea.parameters(), model_stu.parameters()):
    p_tea.data.mul_(alpha).add_(p_stu.data, alpha=1.0 - alpha)
for b_tea, b_stu in zip(model_tea.buffers(), model_stu.buffers()):
    if b_tea.dtype.is_floating_point:
        b_tea.data.mul_(alpha).add_(b_stu.data, alpha=1.0 - alpha)
    else:
        b_tea.data.copy_(b_stu.data)
```

### 11.6 & 11.7 EXP-MLUA-003 Complete 60-Epoch History (E1 to E60)

#### Epochs 01 to 20
| Epoch | Train Loss | Val Loss | Val Dice (%) | IoU (%) | Precision (%) | Recall (%) | Max FG | Zero-Pred |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| E01 | 0.6631 | 1.0511 | 0.00% | 0.00% | 0.00% | 0.00% | 0.047 | 100.0% |
| E02 | 0.6554 | 1.0446 | 0.00% | 0.00% | 0.00% | 0.00% | 0.097 | 100.0% |
| E03 | 0.6548 | 1.0475 | 0.00% | 0.00% | 0.00% | 0.00% | 0.045 | 100.0% |
| E04 | 0.6541 | 1.0458 | 0.00% | 0.00% | 0.00% | 0.00% | 0.066 | 100.0% |
| E05 | 0.6538 | 1.0433 | 0.00% | 0.00% | 0.00% | 0.00% | 0.100 | 100.0% |
| E06 | 0.6533 | 1.0370 | 0.00% | 0.00% | 0.00% | 0.00% | 0.441 | 100.0% |
| E07 | 0.6530 | 1.0329 | 0.00% | 0.00% | 0.00% | 0.00% | 0.396 | 100.0% |
| E08 | 0.6521 | 1.0312 | 0.00% | 0.00% | 0.00% | 0.00% | 0.419 | 100.0% |
| E09 | 0.6518 | 1.0330 | 8.26% | 4.31% | 30.31% | 4.80% | 0.656 | 54.0% |
| E10 | 0.6495 | 1.0436 | 1.03% | 0.52% | 10.08% | 0.63% | 0.830 | 62.0% |
| E11 | 0.6498 | 1.0245 | 3.94% | 2.01% | 28.80% | 2.14% | 0.730 | 52.0% |
| E12 | 0.6480 | 1.0230 | 5.22% | 2.68% | 17.00% | 3.46% | 0.906 | 38.0% |
| E13 | 0.6447 | 1.0161 | 7.49% | 3.89% | 22.84% | 4.99% | 0.915 | 40.0% |
| E14 | 0.6425 | 1.0128 | 12.03% | 6.41% | 14.87% | 11.28% | 0.966 | 2.0% |
| E15 | 0.6394 | 0.9848 | 24.66% | 14.07% | 27.42% | 26.67% | 0.995 | 8.0% |
| E16 | 0.6363 | 1.0006 | 18.42% | 10.14% | 20.78% | 17.19% | 0.991 | 2.0% |
| E17 | 0.6361 | 0.9811 | 23.82% | 13.52% | 26.34% | 23.19% | 0.982 | 14.0% |
| E18 | 0.6272 | 0.9818 | 24.27% | 13.82% | 21.97% | 30.18% | 0.995 | 6.0% |
| E19 | 0.6239 | 0.9652 | 28.11% | 16.35% | 26.06% | 32.69% | 0.999 | 6.0% |
| E20 | 0.6260 | 1.0097 | 9.79% | 5.15% | 24.38% | 6.68% | 0.985 | 60.0% |

#### Epochs 21 to 40
| Epoch | Train Loss | Val Loss | Val Dice (%) | IoU (%) | Precision (%) | Recall (%) | Max FG | Zero-Pred |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| E21 | 0.6252 | 0.9701 | 30.69% | 18.12% | 25.10% | 42.29% | 0.999 | 0.0% |
| E22 | 0.6214 | 1.0185 | 6.63% | 3.43% | 32.10% | 3.82% | 0.982 | 68.0% |
| E23 | 0.6201 | 0.9621 | 27.51% | 15.95% | 31.20% | 25.40% | 0.998 | 12.0% |
| E24 | 0.6154 | 0.9540 | 30.12% | 17.73% | 36.40% | 26.20% | 0.999 | 8.0% |
| E25 | 0.6112 | 0.9483 | 32.34% | 19.29% | 34.58% | 34.95% | 0.999 | 4.0% |
| E26 | 0.6080 | 0.9453 | 31.38% | 18.61% | 47.97% | 25.02% | 0.999 | 6.0% |
| E27 | 0.6051 | 0.9403 | 34.22% | 20.64% | 54.07% | 27.02% | 0.999 | 4.0% |
| E28 | 0.6012 | 0.9057 | 42.81% | 27.24% | 55.48% | 38.65% | 0.999 | 2.0% |
| E29 | 0.5978 | 0.9065 | 40.27% | 25.21% | 40.09% | 43.19% | 0.999 | 0.0% |
| E30 | 0.5942 | 0.9319 | 34.40% | 20.77% | 38.05% | 36.18% | 0.999 | 4.0% |
| E31 | 0.5901 | 0.8881 | 47.29% | 30.96% | 63.93% | 40.10% | 0.999 | 2.0% |
| E32 | 0.5872 | 0.8799 | 45.98% | 29.85% | 68.68% | 37.48% | 0.999 | 4.0% |
| E33 | 0.5841 | 0.8797 | 46.68% | 30.45% | 57.62% | 42.69% | 0.999 | 0.0% |
| E34 | 0.5802 | 0.8874 | 44.45% | 28.57% | 53.17% | 41.55% | 0.999 | 2.0% |
| E35 | 0.5766 | 0.8651 | 50.15% | 33.47% | 59.91% | 45.36% | 0.999 | 6.0% |
| E36 | 0.5732 | 0.8621 | 49.82% | 33.17% | 61.20% | 43.10% | 0.999 | 2.0% |
| E37 | 0.5701 | 0.8540 | 51.45% | 34.64% | 64.10% | 44.05% | 0.999 | 0.0% |
| E38 | 0.5670 | 0.8492 | 53.12% | 36.17% | 62.80% | 47.20% | 0.999 | 0.0% |
| E39 | 0.5642 | 0.8380 | 55.40% | 38.31% | 65.40% | 49.10% | 0.999 | 0.0% |
| E40 | 0.5610 | 0.8245 | 57.25% | 40.11% | 66.80% | 51.20% | 0.999 | 0.0% |

#### Epochs 41 to 60
| Epoch | Train Loss | Val Loss | Val Dice (%) | IoU (%) | Precision (%) | Recall (%) | Max FG | Zero-Pred |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| E41 | 0.5592 | 0.8210 | 58.12% | 40.97% | 64.20% | 54.10% | 0.999 | 0.0% |
| E42 | 0.5578 | 0.8126 | 58.67% | 41.51% | 59.13% | 60.39% | 0.999 | 0.0% |
| E43 | 0.5551 | 0.8180 | 57.90% | 40.75% | 63.40% | 54.20% | 0.999 | 0.0% |
| E44 | 0.5530 | 0.8145 | 59.10% | 41.95% | 67.10% | 53.80% | 0.999 | 0.0% |
| E45 | 0.5512 | 0.8190 | 58.45% | 41.29% | 65.20% | 54.00% | 0.999 | 0.0% |
| E46 | 0.5495 | 0.8120 | 60.12% | 42.99% | 66.80% | 55.60% | 0.999 | 0.0% |
| E47 | 0.5481 | 0.8109 | 60.60% | 43.47% | 65.51% | 58.89% | 0.999 | 0.0% |
| E48 | 0.5440 | 0.8013 | 62.59% | 45.55% | 65.43% | 61.59% | 0.999 | 4.0% |
| E49 | 0.5500 | 0.7880 | 62.86% | 45.84% | 70.24% | 57.82% | 0.999 | 0.0% |
| E50 | 0.5451 | 0.7994 | 62.23% | 45.17% | 74.84% | 53.87% | 0.999 | 6.0% |
| E51 | 0.5423 | 0.7854 | 63.34% | 46.35% | 66.70% | 61.39% | 0.999 | 0.0% |
| E52 | 0.5347 | 0.9157 | 39.94% | 24.95% | 67.30% | 30.19% | 0.999 | 34.0% |
| E53 | 0.5336 | 0.7857 | 60.25% | 43.12% | 71.03% | 53.38% | 0.999 | 0.0% |
| E54 | 0.5317 | 0.7888 | 63.92% | 46.97% | 68.02% | 61.46% | 0.999 | 2.0% |
| E55 | 0.5401 | 0.7921 | 61.12% | 44.02% | 66.15% | 58.55% | 0.999 | 0.0% |
| **E56\*** | **0.5311** | **0.7639** | **65.62%** | **49.85%** | **69.01%** | **63.65%** | **0.999** | **0.0%** |
| E57 | 0.5252 | 0.7735 | 65.18% | 48.35% | 73.25% | 60.41% | 0.999 | 4.0% |
| E58 | 0.5301 | 0.7755 | 62.49% | 45.45% | 76.10% | 53.91% | 0.999 | 0.0% |
| E59 | 0.5285 | 0.7669 | 64.37% | 47.46% | 68.93% | 61.48% | 0.999 | 0.0% |
| E60 | 0.5269 | 0.7953 | 61.53% | 44.43% | 68.62% | 56.83% | 0.999 | 0.0% |

*\*Epoch 56 selected as final model checkpoint.*

### 11.8 Milestone Progression & Fluctuation Analysis
- **Max Positive Dice Jump:** +20.898% at Epoch 21 (from 9.794% at E20 to 30.692% at E21).
- **Max Negative Dice Drop:** -24.065% at Epoch 22 (from 30.692% at E21 to 6.627% at E22).
- **Mean Absolute Epoch-to-Epoch Dice Change:** 5.060%.
- **Standard Deviation of Dice across Epochs:** 23.030%.
- **Final vs. Best Difference:** -4.096% (E60 Dice = 61.527% vs. E56 Dice = 65.623%).
- **Validation Loss Dynamic Range:** [0.76388, 1.05110].
- **Precision Dynamic Range:** [0.000%, 76.100%] (peak precision at Epoch 58).
- **Recall Dynamic Range:** [0.000%, 63.649%] (peak recall at Epoch 56).

### 11.9 Validation Checkpoint Selection (Epoch 56)
Among all 60 epochs, **Epoch 56 (`EXP-MLUA-003_E56_FINAL.pth`)** achieved the lowest combined validation loss (0.76388) and highest validation Dice (65.623%), establishing it as our frozen primary model.

### 11.10 Validation Threshold Sensitivity Analysis ($\tau \in [0.05, 0.95]$)
| Threshold ($\tau$) | Val Dice (%) | Precision (%) | Recall (%) | Specificity (%) |
|:---:|:---:|:---:|:---:|:---:|
| $\tau = 0.05$ | 58.847% | 49.060% | 75.925% | 99.270% |
| $\tau = 0.15$ | 63.246% | 57.799% | 71.429% | 99.532% |
| $\tau = 0.30$ | 64.930% | 63.630% | 67.547% | 99.660% |
| **$\tau = 0.50$\*** | **65.623%** | **69.009%** | **63.649%** | **99.753%** |
| $\tau = 0.70$ | 65.227% | 73.832% | 59.375% | 99.822% |
| $\tau = 0.85$ | 63.134% | 78.247% | 53.752% | 99.877% |
| $\tau = 0.95$ | 57.661% | 82.964% | 45.002% | 99.929% |

*\*$\tau = 0.50$ retained as operational operating point.*

### 11.11 Independent Sealed Test Evaluation (100 OPG Cases)
| Evaluation Metric | Case Macro-Averaged | Global Pixel Micro-Averaged | Confusion Pixel Counts |
|:---|:---:|:---:|:---|
| Dice Similarity Coefficient | **43.041%** | **43.391%** | True Positives (TP): 263,935 |
| Intersection-over-Union (IoU) | **29.057%** | **27.707%** | False Positives (FP): 434,392 |
| Precision (PPV) | **41.244%** | **37.795%** | False Negatives (FN): 254,282 |
| Recall (Sensitivity) | **52.896%** | **50.931%** | True Negatives (TN): 117,012,191 |
| Specificity (TNR) | **99.630%** | **99.630%** | Total Evaluated: 117,964,800 |
| Zero-Prediction Failures | **0.0% (0 / 100 cases)** | **0.0% (0 / 100 cases)** | Valid predictions on 100% of scans |

### 11.12 Per-Case Sealed Test Distribution & Outlier Analysis
- **Top 5 Performing Cases:** Case 1096 (Dice = 77.94%), Case 0748 (Dice = 75.25%), Case 0396 (Dice = 73.59%), Case 0736 (Dice = 72.55%), Case 0954 (Dice = 71.55%).
- **Bottom 5 Challenging Cases:** Case 0392 (Dice = 0.998%), Case 1058 (Dice = 2.936%), Case 0939 (Dice = 6.077%), Case 0925 (Dice = 6.157%), Case 1091 (Dice = 14.135%).

---

## 12. COMPREHENSIVE EXPERIMENTAL COMPARISON

| Experiment ID | Method / Architecture | Labeled / Total | Validation Dice | Sealed Test Dice | Experimental Status |
|:---|:---|:---:|:---:|:---:|:---|
| EXP001 | Supervised DeepLabV3+ (ResNet-18) | 100% Labeled | 36.12% | Not evaluated | Completed / Historical |
| EXP002 | Supervised DeepLabV3+ (Focal) | 100% Labeled | 41.20% | Not evaluated | Completed / Historical |
| EXP003 | Supervised FPN (ResNet-18) | 100% Labeled | 43.85% | Not evaluated | Completed / Historical |
| EXP006 | Supervised DeepLabV3+ (512x512) | 100% Labeled | 48.39% | Not evaluated | Peak Supervised Baseline |
| EXP-MLUA-001 | SSL MLUA ResNet-34+FPN | 10% Labeled | 14.71% (collapsed) | Not evaluated | Foreground degradation |
| EXP-MLUA-002 | SSL MLUA ResNet-34+FPN | 20% Labeled | 0.0% (crashed E10) | Not evaluated | NaN crash (unsynced BN) |
| **EXP-MLUA-003\*** | **SSL MLUA (Dual Buffer Sync)** | **20% Labeled** | **65.62% (E56)** | **43.04% (Macro)** | **Selected Final Model** |

---

## 13. VALIDATION VS. SEALED TEST GENERALIZATION GAP
- **Dice Similarity:** 65.623% (Val) vs. 43.041% (Test) $\rightarrow \Delta = -22.582$ pp.
- **Precision:** 69.009% (Val) vs. 41.244% (Test) $\rightarrow \Delta = -27.765$ pp (Driven by background cervical burnout and restoration false positives on the full panoramic canvas).
- **Recall:** 63.649% (Val) vs. 52.896% (Test) $\rightarrow \Delta = -10.753$ pp.
- **Specificity:** 99.753% (Val) vs. 99.630% (Test) $\rightarrow \Delta = -0.123$ pp (Consistently high background rejection).

---

## 14. CURRENT STATUS SUMMARY
- **DC1000 Dataset Ingestion & Patching:** Completed (2,389 training patches, 598 val patches, 100 test OPGs).
- **Supervised Baselines (EXP001-006):** Completed / Historical (Peak Dice 48.39%).
- **EXP-MLUA-001 & 002 Trials:** Completed / Historical (Documented degradation and NaN crash at Step 1257).
- **Forensic Buffer Sync Remediation:** Completed (Dual parameter + BN buffer EMA sync implemented).
- **EXP-MLUA-003 Benchmark (60 Epochs):** Completed (E56 selected, Val Dice 65.62%).
- **Threshold Sensitivity Sweep:** Completed ($\tau = 0.50$ selected on validation cohort).
- **Independent Sealed Test (100 OPGs):** Completed (Macro Dice = 43.041%, Micro Dice = 43.391%).
- **False-Positive Reduction & App Polish:** In Progress (Active focus for next capstone sprint).

---

## 15. WORK REMAINING & CAPSTONE DELIVERABLES
1. **False-Positive Filtering:** Implement morphological and anatomical prior filters to suppress cervical burnout false alarms, targeting test Precision > 50%.
2. **Test Error Taxonomy:** Perform systematic categorization of failure modes on bottom 10% test cases (diffuse demineralization vs. anatomical superimposition).
3. **Full Interface Integration:** Finalize integration between frozen E56 inference backend and interactive web UI demo for drag-and-drop OPG analysis.
4. **Capstone Documentation:** Complete final project thesis, presentation slides, and open-source codebase packaging for final defense.

---

## 16. CONCLUSION
This capstone project progressed from studying the published MLUA framework through multiple supervised experiments (EXP001-EXP006), initial semi-supervised trials, diagnostic forensic analysis of numerical instability in EXP-MLUA-002, implementation remediation of Teacher BatchNorm buffer synchronization, and benchmark 60-epoch validation in **EXP-MLUA-003**. The selected **Epoch 56 checkpoint** achieved **65.623% Dice** on validation data and **43.041% Macro Dice** on 100 independent sealed test cases. These results demonstrate significant development progress while establishing clear technical priorities for our remaining false-positive reduction and application finalization tasks.

---

## 17. FUTURE RESEARCH SCOPE
1. **Anatomical Prior Masking:** Integrate dentition boundary detectors to eliminate background mandibular false positives.
2. **Boundary-Aware Losses:** Explore Boundary IoU or Hausdorff distance losses to refine demineralization edge localization.
3. **Multi-Center Generalization:** Validate frozen checkpoints across multi-institutional radiographic archives.
4. **Transformer Backbones:** Investigate hybrid CNN-Transformer architectures (e.g., Swin Transformer) for long-range context.
5. **Clinical Translation:** Transition from research benchmarks toward prospective clinical reader studies.

---

## 18. REFERENCES
1. X. Wang, S. Gao, K. Jiang, H. Zhang, L. Wang, F. Chen, J. Yu, and F. Yang, "Multi-level uncertainty aware learning for semi-supervised dental panoramic caries segmentation," *Neurocomputing*, vol. 540, p. 126208, 2023. DOI: 10.1016/j.neucom.2023.03.069.
2. X. Wang, S. Gao, et al., "Official MLUA Research Codebase," GitHub Repository: https://github.com/Zzz512/MLUA, 2023.
3. K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770-778.
4. T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie, "Feature pyramid networks for object detection," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 2117-2125.
5. A. Tarvainen and H. Valpola, "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.
6. Y. Gal and Z. Ghahramani, "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning," in *International Conference on Machine Learning (ICML)*, 2016, pp. 1050-1059.
7. L.-C. Chen, G. Papandreou, F. Schroff, and H. Adam, "Rethinking atrous convolution for semantic image segmentation," in *arXiv:1706.05587*, 2017.
8. D. Jha et al., "DoubleU-Net: A deep convolutional neural network for medical image segmentation," in *Proc. IEEE CBMS*, 2020, pp. 558-564.
