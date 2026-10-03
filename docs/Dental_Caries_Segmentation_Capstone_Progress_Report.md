# B.TECH CAPSTONE PROJECT PROGRESS REPORT

# Progress Report on Dental Caries Segmentation from Panoramic X-Ray Images Using Multi-Level Uncertainty-Aware Learning

**A Study on Baseline Implementation, Training Diagnostics, Buffer Remediation, and Empirical Validation on the DC1000 Benchmark**

---

### METADATA
- **Student Name:** Nupur Madaan
- **Degree Program:** Bachelor of Technology (B.Tech) in Computer Science & Engineering
- **Enrollment / Roll No.:** [Placeholder: Student Enrollment Number]
- **Department:** [Placeholder: Department of Computer Science & Engineering]
- **Institution / University:** [Placeholder: University / Institute Name]
- **Faculty Guide / Supervisor:** [Placeholder: Faculty Supervisor / Guide Name]
- **Project Domain:** Medical Computer Vision & Deep Learning (Dental Imaging)
- **Current Major Milestone:** EXP-MLUA-003 (60-Epoch Remediated Run & Sealed Evaluation Completed)
- **Academic Year:** 2026-27

---

> **Progress Narrative Summary:** This progress report documents the progressive engineering and experimental journey of our capstone project. Starting from the published baseline paper (Wang et al., *Neurocomputing* 2023), we implemented the semi-supervised Teacher-Student framework, investigated early training degradation (EXP-001) and numerical collapse (EXP-002), isolated the root cause to Teacher BatchNorm buffer desynchronization, remediated the update pipeline in EXP-003, validated the resulting model (Epoch 56 Val Dice = 65.62%), completed independent sealed testing (Macro Dice = 43.04%), and identified the specific remaining tasks required for capstone completion.

---

## TABLE OF CONTENTS
1. [ABSTRACT](#1-abstract)
2. [INTRODUCTION](#2-introduction)
3. [OBJECTIVES](#3-objectives)
4. [PROBLEM STATEMENT](#4-problem-statement)
5. [PROJECT PROGRESS / WORK COMPLETED](#5-project-progress--work-completed)
6. [FLOWCHART & SYSTEM ARCHITECTURE](#6-flowchart--system-architecture)
7. [METHODOLOGY](#7-methodology)
   - 7.1 Input Collection & DC1000 Benchmark
   - 7.2 Dataset Characteristics & Binary Segmentation Scope
   - 7.3 Data Preparation & 20/80 Partitioning
   - 7.4 Preprocessing & Normalization
   - 7.5 384x384 Patch Extraction & Sliding Windows
   - 7.6 ResNet-34 Feature Extraction Backbone
   - 7.7 Feature Pyramid Network (FPN) Decoder
   - 7.8 Multi-Scale Auxiliary & Fused Segmentation Heads
   - 7.9 Teacher-Student Consistency Learning
   - 7.10 Loss Functions (BCE + Soft Dice + Consistency)
   - 7.11 Teacher EMA Update & Buffer Remediation
   - 7.12 Numerical Instability Analysis (EXP-MLUA-002)
   - 7.13 Monte Carlo Uncertainty Estimation & Confidence Masking
   - 7.14 Threshold Selection (Validation Sweep)
   - 7.15 Sliding-Window Panoramic Reconstruction
   - 7.16 Final Output Representation
8. [SYSTEM IMPLEMENTATION MODULES](#8-system-implementation-modules)
9. [EXPERIMENTAL PROGRESS AND RESULTS](#9-experimental-progress-and-results)
   - 9.1 Published MLUA Baseline Context
   - 9.2 EXP-MLUA-001 (Baseline Run & Degradation)
   - 9.3 EXP-MLUA-002 (Numerical Instability & Collapse)
   - 9.4 Forensic Diagnostic Analysis of EXP-MLUA-002
   - 9.5 Remediation: Dual Parameter & Buffer EMA Synchronization
   - 9.6 EXP-MLUA-003 (Corrected 60-Epoch Run)
   - 9.7 Best Checkpoint Selection (Epoch 56)
   - 9.8 Validation Threshold Sensitivity Analysis
   - 9.9 Independent Sealed Test Evaluation (100 OPGs)
   - 9.10 Validation vs. Sealed Test Generalization Gap
10. [CURRENT STATUS AND WORK REMAINING](#10-current-status-and-work-remaining)
11. [CONCLUSION](#11-conclusion)
12. [FUTURE SCOPE](#12-future-scope)
13. [REFERENCES](#13-references)

---

## 1. ABSTRACT
Automated localization and segmentation of dental caries in orthopantomograms (panoramic dental X-rays) is a critical task in digital oral healthcare. However, developing reliable deep learning segmentation models for panoramic radiographs is severely hindered by extreme class imbalance (lesions occupy <0.5% of total image area), low local radiographic contrast, diffuse demineralization boundaries, anatomical superimpositions, and the prohibitive cost of expert manual annotations. To address clinical annotation scarcity, this capstone project investigates semi-supervised learning based on Multi-Level Uncertainty Aware (MLUA) Teacher-Student consistency architectures using the public DC1000 dataset (1,000 panoramic radiographs).

This progress report outlines our complete development journey, from studying the original MLUA publication (Wang et al., *Neurocomputing* 2023) to conducting sequential experimental trials. In our initial experiments, the baseline model exhibited training degradation and severe foreground suppression (EXP-MLUA-001). Subsequent scaling in EXP-MLUA-002 encountered catastrophic numerical instability (NaN loss collapse during Teacher forward passes). A systematic forensic audit isolated the root cause: during Exponential Moving Average (EMA) updates, Teacher model weights were updated while internal BatchNorm running statistics remained un-synchronized at default values, leading to explosive GroupNorm activations. We resolved this issue by implementing dual parameter and buffer EMA synchronization in **EXP-MLUA-003**, which successfully completed 60 epochs without instability.

The remediated model achieved a peak **Validation Dice of 65.623%**, **IoU of 49.854%**, **Precision of 69.009%**, and **Recall of 63.649%** at **Epoch 56** with an operating threshold of $\tau = 0.50$ on 384×384 patches. When evaluated on an independent, frozen sealed test set of 100 full panoramic images, the model achieved a **Macro Dice of 43.041%**, **Macro Recall of 52.896%**, and **Specificity of 99.630%** (Micro Dice = 43.391%). The observed generalization gap highlights key remaining challenges, including background false-positive reduction on cervical burnout artifacts and boundary calibration. Current work is focused on error analysis, post-processing refinement, and final application interface integration.

---

## 2. INTRODUCTION
Dental caries is among the most widespread chronic oral diseases globally. Left untreated, bacterial acid demineralization penetrates the enamel-dentin junction into the dental pulp, causing severe pain, periapical infection, and tooth loss. Orthopantomography (panoramic dental radiography) is the most widely utilized screening modality in clinical dentistry because a single scan captures the entire maxillofacial region, including all maxillary and mandibular teeth, supporting alveolar bone, and adjacent anatomical structures.

Despite its clinical ubiquity, automated caries detection on panoramic X-rays presents substantial computer vision challenges:
- **Extreme Class Imbalance:** Carious demineralization regions occupy minute spatial areas relative to the full panoramic canvas (often less than 0.5% of total pixels).
- **Ambiguous Attenuation & Diffuse Margins:** Incipient caries appears as subtle radiolucency with gradual, ill-defined boundaries rather than sharp edges.
- **Confounding Anatomical Artifacts:** Normal radiographic phenomena, such as cervical burnout at the tooth neck, overlapping proximal tooth contours, and ghost images from contralateral bones, closely mimic true carious lesions.
- **Annotation Scarcity:** Creating dense, pixel-level ground truth contours requires hours of labor by expert dental clinicians, making large fully supervised datasets difficult to obtain.

To address these challenges, our project explores semi-supervised deep learning inspired by the Multi-Level Uncertainty Aware (MLUA) framework. Semi-supervised learning utilizes a small fraction of labeled images alongside a larger collection of unlabeled images. By employing a Teacher-Student consistency model with Monte Carlo uncertainty estimation, the framework generates pseudo-labels for unlabeled regions while dynamically filtering out unreliable predictions.

This report presents our project's progressive milestones, detailing how initial experimental failures were analyzed and systematically resolved to establish a functional, validated segmentation pipeline.

---

## 3. OBJECTIVES
1. Study the foundational MLUA semi-supervised learning framework proposed for dental panoramic caries segmentation.
2. Implement the ResNet-34 encoder and Feature Pyramid Network (FPN) decoder architecture on the benchmark DC1000 dataset.
3. Investigate model behavior and training dynamics across progressive experimental iterations (EXP-MLUA-001, 002, and 003).
4. Diagnose and resolve the numerical instability (NaN loss) encountered in Teacher forward passes during early training.
5. Correct the Teacher Exponential Moving Average (EMA) update routine by incorporating strict BatchNorm buffer synchronization.
6. Evaluate the corrected model over 60 epochs to select the optimal validation checkpoint (Epoch 56).
7. Conduct a validation threshold sensitivity sweep ($\tau \in [0.05, 0.95]$) to establish the optimal operating cutoff ($\tau = 0.50$).
8. Perform independent evaluation on a frozen sealed test set of 100 panoramic images to assess real-world generalization.
9. Document the observed generalization gap and define the remaining development tasks required for capstone completion.

---

## 4. PROBLEM STATEMENT
**Formal Problem Definition:** Given a digitized panoramic dental X-ray image $X \in \mathbb{R}^{H \times W}$, the objective is to develop a deep learning system that outputs a pixel-level probability map $P(Y = 1 | X)$ representing suspected carious demineralization, and converts it into a binary segmentation mask $\hat{Y}_\tau \in \{0, 1\}^{H \times W}$ at threshold $\tau = 0.50$. The model must maximize overlap with expert consensus annotations while maintaining robust numerical stability and operating under limited labeled training data (20% labeled, 80% unlabeled).

---

## 5. PROJECT PROGRESS / WORK COMPLETED

| Development Phase | Work Performed | Key Result / Observation | Status |
|:---|:---|:---|:---:|
| 1. Literature Study | Analyzed Wang et al. (2023) MLUA paper and official repository. | Understood Teacher-Student SSL and MC uncertainty principles. | Completed |
| 2. Data Preparation | Standardized DC1000 images (768×1536) & extracted 384×384 patches. | Prepared 530 labeled and 1,859 unlabeled patches (20/80 split). | Completed |
| 3. Baseline Implementation | Constructed PyTorch ResNet-34 + FPN pipeline with auxiliary heads. | Verified data loading, forward passes, and basic loss modules. | Completed |
| 4. EXP-MLUA-001 | Conducted initial baseline training run on DC1000 patches. | Observed severe foreground suppression and sub-optimal Dice. | Completed |
| 5. EXP-MLUA-002 | Scaled consistency weighting and loss hyper-parameters. | Encountered fatal NaN loss collapse in Teacher GroupNorm. | Completed |
| 6. Forensic Audit | Audited intermediate activations, layer stats, and buffer states. | Isolated missing Teacher BatchNorm buffer synchronization during EMA. | Completed |
| 7. Buffer Remediation | Implemented dual parameter & buffer EMA update in training loop. | Pre-flight testing confirmed stable, bounded feature activations. | Completed |
| 8. EXP-MLUA-003 | Executed 60-epoch remediated training run (7,920 global steps). | Training completed smoothly; peak Val Dice = 65.62% at Epoch 56. | Completed |
| 9. Threshold Analysis | Evaluated validation threshold sweep from $\tau = 0.05$ to 0.95. | $\tau = 0.50$ identified as optimal validation operating cutoff. | Completed |
| 10. Sealed Testing | Evaluated frozen E56 model on 100 independent sealed OPGs. | Achieved Macro Dice = 43.04%, Micro Dice = 43.39%, Spec = 99.63%. | Completed |
| 11. Application Integration | Designed inference pipeline and sliding-window reconstruction. | Backend reconstruction verified; UI polishing underway. | In Progress |

---

## 6. FLOWCHART & SYSTEM ARCHITECTURE

```
+----------------------------------------------------------------------------------------------------+
|                                  PROJECT PROGRESS FLOWCHART                                        |
+----------------------------------------------------------------------------------------------------+
|  1. MLUA Paper & DC1000  --->  2. Baseline Implementation  --->  3. EXP-MLUA-001 (Degradation)     |
|                                                                                |                   |
|  6. Buffer Remediation   <---  5. Forensic Investigation  <---  4. EXP-MLUA-002 (NaN Collapse)     |
|          |                                                                                         |
|          v                                                                                         |
|  7. EXP-MLUA-003 (60 Ep) --->  8. Threshold & Sealed Test --->  9. Remaining Work (FP Reduction)  |
+----------------------------------------------------------------------------------------------------+
```

---

## 7. METHODOLOGY

### 7.1 Input Collection & DC1000 Benchmark
The project uses the publicly available **DC1000** dental caries dataset, containing 1,000 digitized panoramic radiographs acquired in clinical settings. The dataset provides expert annotations for over 7,500 individual carious lesions, categorized into detailed (593 cases) and rough (407 cases) annotation subsets.

### 7.2 Dataset Characteristics & Binary Segmentation Scope
While the original dataset documentation mentions clinical caries depth levels (shallow, middle, deep), our current system is designed strictly for **pixel-level binary semantic segmentation** ($0 = \text{Background / Healthy}, 1 = \text{Suspected Dental Caries}$). This focus ensures maximum spatial localization accuracy without imposing multi-class boundary ambiguities.

### 7.3 Data Preparation & 20/80 Partitioning
To simulate realistic clinical annotation constraints, the training dataset was partitioned into **20% labeled data (530 patches)** and **80% unlabeled data (1,859 patches)**, totaling 2,389 training patches. An independent validation set of 598 patches and a sealed test set of 100 full panoramic images were preserved with strict patient-level separation.

### 7.4 Preprocessing & Normalization
Raw radiographs with varying dimensions were standardized to $768\times 1536$ pixels. Pixel intensities were scaled to $[0, 1]$ and normalized using standard ImageNet mean and variance constants ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

### 7.5 384×384 Patch Extraction & Sliding Windows
To preserve fine lesion textures without exceeding GPU memory capacity, panoramic images were divided into **384×384 pixel patches** using a sliding window with vertical and horizontal stride $S = 192$ (50% overlap), yielding 21 patches per panoramic image.

### 7.6 ResNet-34 Feature Extraction Backbone
The encoder uses a deep ResNet-34 architecture pre-trained on ImageNet. Its residual connections facilitate gradient propagation, generating multi-scale feature hierarchies across stages $C_2$ (64 ch), $C_3$ (128 ch), $C_4$ (256 ch), and $C_5$ (512 ch).

### 7.7 Feature Pyramid Network (FPN) Decoder
The FPN decoder combines deep semantic features with shallow spatial details. Lateral 1×1 convolutions unify channel dimensions to 256, and top-down pathways upsample and merge features to create pyramid levels $P_2, P_3, P_4,$ and $P_5$.

### 7.8 Multi-Scale Auxiliary & Fused Segmentation Heads
Four auxiliary convolution heads output intermediate segmentation logits from $P_2-P_5$, providing deep supervision during training. A primary fused head concatenates and upsamples all pyramid levels to generate the final $384\times 384$ prediction.

### 7.9 Teacher-Student Consistency Learning
The system maintains two parallel network instances: the Student network (updated via backpropagation) and the Teacher network (updated via Exponential Moving Average of Student weights). The Teacher generates pseudo-labels for unlabeled images, driving consistency learning.

### 7.10 Loss Functions (BCE + Soft Dice + Consistency MSE)
On labeled patches, the supervised loss is a combination of Binary Cross-Entropy (BCE) and Soft Dice loss: $\mathcal{L}_{sup} = \mathcal{L}_{BCE} + \mathcal{L}_{Dice}$. On unlabeled patches, an unsupervised Mean Squared Error (MSE) consistency loss is computed between Student predictions and Teacher pseudo-labels.

### 7.11 Teacher EMA Update & Buffer Remediation
Teacher weights are updated with momentum $\alpha = 0.99$: $\theta_T = \alpha \theta_T + (1 - \alpha) \theta_S$. In our remediated implementation, Teacher BatchNorm buffers (running mean and variance) are also explicitly synchronized with Student buffers at each step, preventing activation divergence.

### 7.12 Numerical Instability Analysis (EXP-MLUA-002)
During EXP-MLUA-002, training failed due to NaN values in Teacher GroupNorm layers. Forensic investigation revealed that because the Teacher ran in eval mode, its BatchNorm running buffers were never updated. When combined with EMA-updated weights, this caused internal activations to explode to extreme values (~$10^4$), overflowing numerical precision. Synchronizing buffers in EXP-MLUA-003 fully eliminated this failure.

### 7.13 Monte Carlo Uncertainty Estimation & Confidence Masking
For each unlabeled patch, the Teacher performs $T = 8$ stochastic forward passes under dropout/perturbation. Predictive uncertainty is estimated as pixel-wise variance. A binary confidence mask $M(i, j)$ filters out high-uncertainty pixels from the consistency loss, preventing the Student from learning noisy pseudo-labels.

### 7.14 Threshold Selection (Validation Sweep)
A validation-only sensitivity sweep evaluated decision thresholds from $\tau = 0.05$ to $0.95$. Threshold **$\tau = 0.50$** yielded the highest validation Dice (65.623%) with balanced Precision and Recall, and was retained as the frozen operating threshold.

### 7.15 Sliding-Window Panoramic Reconstruction
During full-image inference, 21 overlapping 384×384 patch predictions are stitched back into the $768\times 1536$ coordinate grid, with overlapping areas averaged via linear blending.

### 7.16 Final Output Representation
The system outputs a continuous caries probability map, a thresholded binary mask ($\tau = 0.50$), and bounding coordinates highlighting suspected carious regions for clinical review.

---

## 8. SYSTEM IMPLEMENTATION MODULES

| Module Name | Implementation File | Core Functional Responsibility |
|:---|:---|:---|
| 8.1 Dataset & Loader | `src/mlua/data/dataset.py` | Loads DC1000 OPGs, applies normalization, extracts 384x384 patches. |
| 8.2 Patch Generator | `src/mlua/data/patch_extractor.py` | Extracts 21 patches per 768x1536 image with stride S=192. |
| 8.3 ResNet-34 Encoder | `src/mlua/models/encoder.py` | Extracts hierarchical visual features across stages C2-C5. |
| 8.4 FPN Decoder | `src/mlua/models/fpn.py` | Fuses multi-scale features; hosts 4 auxiliary + 1 fused heads. |
| 8.5 Teacher-Student Engine | `src/mlua/models/teacher_student.py` | Manages dual models, EMA weight updates, and buffer sync. |
| 8.6 Uncertainty Estimator | `src/mlua/uncertainty/mc_sampler.py` | Performs T=8 Monte Carlo passes; generates confidence masks. |
| 8.7 Loss Manager | `src/mlua/losses/combined_loss.py` | Computes BCE, Soft Dice, and masked consistency MSE losses. |
| 8.8 Training Trainer | `src/mlua/trainers/trainer.py` | Orchestrates 60-epoch loop, AdamW optimizer, and Cosine scheduler. |
| 8.9 Inference & Stitching | `src/mlua/inference/reconstruction.py` | Executes sliding-window inference and blends full OPG masks. |
| 8.10 Evaluation Metrics | `src/mlua/metrics/evaluator.py` | Computes macro and micro Dice, IoU, Precision, Recall, Specificity. |

---

## 9. EXPERIMENTAL PROGRESS AND RESULTS

### 9.1 Published MLUA Baseline Context
The baseline paper (Wang et al., *Neurocomputing* 2023) established that multi-level uncertainty estimation can effectively guide semi-supervised segmentation on panoramic dental radiographs. We adopted this core concept while adapting the implementation to our standardized DC1000 dataset partition and modern PyTorch environment.

### 9.2 EXP-MLUA-001 (Baseline Run & Degradation)
Our first experiment, **EXP-MLUA-001**, evaluated the initial baseline setup. While training completed, the model suffered from foreground suppression, predicting very few caries pixels due to extreme background dominance (validation Dice ~48%). This indicated the need for improved loss balancing and consistency scaling.

### 9.3 EXP-MLUA-002 (Numerical Instability & Collapse)
In **EXP-MLUA-002**, we scaled the consistency weight and refined patch augmentations. However, training collapsed abruptly in early epochs with non-finite (NaN) loss values during the Teacher forward pass.

### 9.4 Forensic Diagnostic Analysis of EXP-MLUA-002
A deep diagnostic audit traced the NaN failure to the Teacher network's GroupNorm layers. Because the Teacher was evaluated in eval mode, its BatchNorm running buffers were never updated during training. Meanwhile, its weights were continuously updated via EMA from the Student. This disparity between weights and uninitialized buffer statistics caused internal activations to explode to extreme values (~$10^4$), overflowing numerical precision.

### 9.5 Remediation: Dual Parameter & Buffer EMA Synchronization
To remediate the instability, we modified the EMA update routine to synchronize both model parameters and BatchNorm running statistics:

```python
for t_buf, s_buf in zip(teacher.buffers(), student.buffers()):
    t_buf.data.copy_(alpha * t_buf.data + (1.0 - alpha) * s_buf.data)
```

### 9.6 EXP-MLUA-003 (Corrected 60-Epoch Run)
With buffer synchronization implemented, **EXP-MLUA-003** completed 60 epochs (7,920 global steps) without any numerical instability.

| Epoch (Step) | Val Loss | Val Dice (%) | Precision (%) | Recall (%) | Observation / Note |
|:---|:---:|:---:|:---:|:---:|:---|
| Epoch 01 (Step 132) | 1.2415 | 41.21% | 45.12% | 37.93% | Initial baseline convergence |
| Epoch 20 (Step 2640) | 0.9124 | 57.84% | 61.23% | 54.81% | Steady progressive learning |
| Epoch 40 (Step 5280) | 0.8045 | 63.12% | 66.45% | 60.10% | Consistent boundary delineation |
| Epoch 50 (Step 6600) | 0.7782 | 64.89% | 68.12% | 62.02% | High precision stability |
| **Epoch 56 (Step 7392)\*** | **0.7639** | **65.62%** | **69.01%** | **63.65%** | **Optimal peak validation checkpoint** |
| Epoch 60 (Step 7920) | 0.7712 | 64.92% | 68.21% | 61.94% | Slight post-peak plateau |

*\*Epoch 56 selected as final model checkpoint.*

### 9.7 Best Checkpoint Selection (Epoch 56)
Among all evaluated epochs, **Epoch 56 (`EXP-MLUA-003_E56_FINAL.pth`)** achieved the lowest validation loss (0.76388) and highest validation Dice (65.623%), establishing it as our frozen primary model.

### 9.8 Validation Threshold Sensitivity Analysis
Using checkpoint E56, a threshold sensitivity sweep across $\tau \in [0.05, 0.95]$ confirmed that **$\tau = 0.50$** provides the optimal trade-off between Precision and Recall on validation data.

| Threshold ($\tau$) | Val Dice (%) | Threshold ($\tau$) | Val Dice (%) | Threshold ($\tau$) | Val Dice (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $\tau = 0.10$ | 61.86% | $\tau = 0.40$ | 65.42% | $\tau = 0.70$ | 65.23% |
| $\tau = 0.20$ | 64.02% | **$\tau = 0.50$\*** | **65.62%** | $\tau = 0.80$ | 64.17% |
| $\tau = 0.30$ | 64.93% | $\tau = 0.60$ | 65.52% | $\tau = 0.90$ | 61.40% |

### 9.9 Independent Sealed Test Evaluation (100 OPGs)
The frozen E56 model was evaluated on an independent sealed test set of 100 full panoramic images (117,964,800 pixels) using sliding-window reconstruction at $\tau = 0.50$. The model achieved **Macro Dice = 43.041%**, **Macro IoU = 29.057%**, **Macro Precision = 41.244%**, **Macro Recall = 52.896%**, and **Macro Specificity = 99.630%**, with **Micro Dice = 43.391%** (TP = 263,935, FP = 434,392, FN = 254,282, TN = 117,012,191).

### 9.10 Validation vs. Sealed Test Generalization Gap

| Performance Metric | Validation (E56 Patches) | Sealed Test (Macro OPG) | Difference ($\Delta$) | Primary Diagnostic Cause |
|:---|:---:|:---:|:---:|:---|
| Dice Similarity | 65.623% | 43.041% | -22.582 pp | Full-image background artifacts & stitching |
| Intersection-over-Union | 49.854% | 29.057% | -20.797 pp | Compounded boundary penalty on OPG canvas |
| Precision | 69.009% | 41.244% | -27.765 pp | Cervical burnout & restoration false positives |
| Recall | 63.649% | 52.896% | -10.753 pp | Incipient non-cavitated demineralization misses |
| Specificity | 99.753% | 99.630% | -0.123 pp | Consistently high background rejection |

---

## 10. CURRENT STATUS AND WORK REMAINING

### Table 6: Current Component Implementation Status
| System Component / Milestone | Current Implementation Status | Verification Artifact / Evidence |
|:---|:---:|:---|
| DC1000 Dataset Preparation | Completed | 2,389 training patches (530 L / 1,859 U) |
| Baseline MLUA Implementation | Completed | `src/mlua/` codebase functional |
| EXP-MLUA-001 & 002 Trials | Completed / Historical | Documented degradation and NaN crash |
| Forensic Buffer Remediation | Completed | Dual weight + BN buffer EMA sync implemented |
| EXP-MLUA-003 Training (60 Epochs) | Completed | `outputs/experiments/EXP-MLUA-003_FINAL/` |
| Validation Checkpoint Selection | Completed (Epoch 56) | Val Dice = 65.623% @ $\tau = 0.50$ |
| Validation Threshold Sensitivity | Completed | 19 thresholds evaluated ($\tau = 0.05$ to 0.95) |
| Sealed Test Evaluation (100 OPGs) | Completed | Macro Dice = 43.041%, Micro = 43.391% |
| Full-Image Inference Pipeline | Completed | Sliding-window reconstruction verified |
| Application UI / Interface | In Progress | Web UI demo active in repository |
| False-Positive Reduction & Tuning | Pending / In Progress | Target for next capstone sprint |

### Table 7: Remaining Capstone Project Deliverables
| Remaining Task | Detailed Technical Scope | Target Objective / Metric |
|:---|:---|:---|
| 1. False-Positive Filtering | Implement post-processing morphological filters to remove cervical burnout artifacts. | Improve sealed test Precision from 41.2% to >50%. |
| 2. Test Error Categorization | Conduct systematic error auditing on the lowest 10% test cases to isolate failure modes. | Identify dominant anatomical failure patterns. |
| 3. UI / Application Polish | Complete seamless integration between frozen E56 backend and interactive front-end. | Enable real-time drag-and-drop OPG analysis. |
| 4. Final Documentation | Compile final capstone report, code documentation, and presentation slide deck. | Final project defense readiness. |

---

## 11. CONCLUSION
This capstone project began with the study and baseline implementation of the published MLUA semi-supervised learning approach for dental panoramic caries segmentation. Initial experimental trials revealed significant challenges: EXP-MLUA-001 showed severe foreground degradation, while EXP-MLUA-002 encountered catastrophic numerical instability during Teacher inference. A thorough forensic investigation identified the absence of Teacher BatchNorm buffer synchronization during EMA updates as the core issue. Correcting this in **EXP-MLUA-003** enabled a stable 60-epoch training run. The resulting **Epoch 56 checkpoint** achieved **65.623% Dice** on validation data and **43.041% Macro Dice** (52.896% Recall, 99.630% Specificity) on an independent sealed test set of 100 full panoramic images. These results establish substantial progress while clearly delineating the remaining work required in false-positive reduction, error analysis, and application finalization.

---

## 12. FUTURE SCOPE
1. **Anatomical False-Positive Filtering:** Integrate dedicated tooth-boundary segmentation priors to mask out non-dentition background areas.
2. **Boundary-Aware Loss Regularization:** Explore boundary-weighted loss functions (e.g., Boundary IoU or Hausdorff loss) to sharpen diffuse demineralization contours.
3. **Multi-Center Cross-Scanner Validation:** Evaluate model robustness across radiographs from different imaging hardware and clinical institutions.
4. **Transformer-Based Encoders:** Investigate Swin Transformer backbones to capture global bilateral dental arch symmetry.
5. **Interactive Clinical Workflow:** Develop an interactive review interface enabling dental practitioners to adjust sensitivity thresholds in real time.

---

## 13. REFERENCES
1. X. Wang, S. Gao, K. Jiang, H. Zhang, L. Wang, F. Chen, J. Yu, and F. Yang, "Multi-level uncertainty aware learning for semi-supervised dental panoramic caries segmentation," *Neurocomputing*, vol. 540, p. 126208, 2023. DOI: 10.1016/j.neucom.2023.03.069.
2. X. Wang, S. Gao, et al., "Official MLUA Research Codebase," GitHub Repository: https://github.com/Zzz512/MLUA, 2023.
3. K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770-778.
4. T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie, "Feature pyramid networks for object detection," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 2117-2125.
5. A. Tarvainen and H. Valpola, "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.
6. Y. Gal and Z. Ghahramani, "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning," in *International Conference on Machine Learning (ICML)*, 2016, pp. 1050-1059.
7. F. Milletari, N. Navab, and S.-A. Ahmadi, "V-Net: Fully convolutional neural networks for volumetric medical image segmentation," in *Fourth International Conference on 3D Vision (3DV)*, 2016, pp. 565-571.
8. I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *International Conference on Learning Representations (ICLR)*, 2019.
