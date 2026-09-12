# MLUA ABLATION STUDY PLAN & PROTOCOL
**Project**: Dental Panoramic Caries Segmentation (MLUA)  
**Document**: `docs/ABLATION_STUDY_PLAN.md`  
**Scientific Setting**: Paper-Aligned Multi-Disturbance Ablation Study (8-Arm Matrix)  
**Execution Status**: PREPARATION & AUDIT ONLY (Training NOT Started)

---

## 1. Scientific Purpose & Hypotheses

The official MLUA paper introduces three core disturbance mechanisms to enable effective semi-supervised learning under high uncertainty:
1. **Iterative Disturbance ($\mathcal{D}_{\text{iter}}$)**: Exponential Moving Average (EMA) teacher ($\alpha=0.99$), enforcing temporal consistency across optimization steps.
2. **Noisy Disturbance ($\mathcal{D}_{\text{noise}}$)**: Additive Gaussian perturbation on teacher input ($\sigma=0.01$, clamp $[-0.1, 0.1]$) combined with Monte Carlo dropout sampling ($T=8$) and dynamic entropy thresholding ($E_{\text{th}} = b \cdot (1 - e/E_{\text{max}}) \cdot \ln(2) + 1.0 \cdot (e/E_{\text{max}}) \cdot \ln(2)$).
3. **Multi-Scale Disturbance ($\mathcal{D}_{\text{scale}}$)**: Multi-level feature pyramid supervision across 4 auxiliary decoder stages ($P_{\text{aux}, 1..4}$) plus the fused output head ($P_{\text{fused}}$), incorporating multi-scale feature ensembles into the uncertainty assessment.

### Core Hypotheses to Test:
- **H1 (Individual Contribution)**: Does adding each individual disturbance mechanism ($\text{ABL-01}, \text{ABL-02}, \text{ABL-03}$) over the unregularized baseline ($\text{ABL-00}$) yield measurable gains in validation Dice and caries sensitivity?
- **H2 (Pairwise Synergy)**: Does combining two disturbance mechanisms ($\text{ABL-04}, \text{ABL-05}, \text{ABL-06}$) produce synergistic regularizing effects superior to any individual disturbance?
- **H3 (Tri-Disturbance MLUA Optimality)**: Does the full tri-disturbance MLUA framework ($\text{ABL-07}$) achieve the lowest prediction variance, highest boundary recall, and best overall Dice coefficient on dental panoramic caries detection?

---

## 2. 8-Arm Ablation Matrix

| Experiment ID | Configuration Name | Iterative ($\mathcal{D}_{\text{iter}}$) | Noisy ($\mathcal{D}_{\text{noise}}$) | Multi-scale ($\mathcal{D}_{\text{scale}}$) | Scientific Goal |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`ABL-00`** | Baseline | **OFF** | **OFF** | **OFF** | Baseline SSL without MLUA disturbances |
| **`ABL-01`** | Iterative Only | **ON** | **OFF** | **OFF** | Measure temporal EMA smoothing alone |
| **`ABL-02`** | Noisy Only | **OFF** | **ON** | **OFF** | Measure MC uncertainty + noise filtering alone |
| **`ABL-03`** | Multi-scale Only | **OFF** | **OFF** | **ON** | Measure deep supervision alone |
| **`ABL-04`** | Iterative + Noisy | **ON** | **ON** | **OFF** | Measure temporal + noise synergy (No deep supervision) |
| **`ABL-05`** | Iterative + Multi-scale | **ON** | **OFF** | **ON** | Measure temporal + multi-scale synergy (No MC noise) |
| **`ABL-06`** | Noisy + Multi-scale | **OFF** | **ON** | **ON** | Measure noise + multi-scale synergy (No EMA teacher) |
| **`ABL-07`** | Full MLUA | **ON** | **ON** | **ON** | Full tri-disturbance MLUA system |

---

## 3. Controlled Variables & Scientific Invariance

To guarantee strict scientific validity, every other parameter across all 8 ablations is frozen identically:

| Variable | Controlled Value | Rationale |
| :--- | :--- | :--- |
| **Dataset Pool** | DC1000 Training Patches (2,389 total) | Identical image collection |
| **Split & Seed** | Seed 42: 265 Labeled (10%) / 2,124 Unlabeled (90%) | Paper-aligned 10% SSL benchmark |
| **Patch Resolution** | $384 \times 384$ pixels (Grayscale 1-channel) | Standardized patch geometry |
| **Batch Composition** | Batch size = 8 ($4\text{ Labeled} + 4\text{ Unlabeled}$) | `TwoStreamBatchSampler` (465 steps/epoch) |
| **Encoder Backbone** | ResNet-34 (ImageNet pretrained initialization) | Exact architecture parity |
| **Optimizer** | AdamW ($\text{lr}=10^{-3}$, $\text{wd}=10^{-2}$, $\beta=(0.9, 0.999)$) | Frozen optimizer policy |
| **LR Scheduler** | Polynomial Decay ($\text{power}=0.9$, $\text{max\_epochs}=200$) | Frozen decay schedule |
| **Consistency Weight** | Sigmoid ramp-up to $w_{\text{max}}=0.10$ over 200 epochs | Standardized regularization curve |
| **Validation Protocol** | Deterministic panoramic reconstruction (21 patches, stride 192) | Zero random augmentation |
| **Decision Threshold** | Fixed at $\tau = 0.50$ | Unbiased evaluation standard |
| **Sealed Test Set** | `dataset/test/` (100 cases) | 100% untouched and sealed |

---

## 4. Metric Reservation & Logging Schema

### Primary Paper-Aligned Metrics
- **Dice Coefficient (F1-Score)**: Primary segmentation overlap metric.
- **Sensitivity / Recall**: Primary clinical safety metric (caries detection rate).
- **Precision**: Positive predictive value (false positive control).

### Additional Diagnostic Metrics
- **Intersection over Union (IoU / Jaccard Index)**
- **Specificity**: True negative rate over healthy enamel/dentin tissue.
- **Accuracy**: Overall pixel classification accuracy.
- **Predicted Foreground Prevalence**: Percentage of positive caries pixels predicted.
- **Max Foreground Probability**: Network confidence peak per epoch.

---

## 5. Comparison Output Schema

All experimental outcomes will be aggregated into [`outputs/experiments/ABLATION_COMPARISON.csv`](file:///c:/Users/devin/MLUA/outputs/experiments/ABLATION_COMPARISON.csv):

```csv
Experiment,Iterative,Noisy,Multi-scale,Dice,Sensitivity,Precision,IoU,F1,Specificity
ABL-00,OFF,OFF,OFF,NOT_RUN,NOT_RUN,NOT_RUN,NOT_RUN,NOT_RUN,NOT_RUN
ABL-01,ON,OFF,OFF,NOT_RUN,NOT_RUN,NOT_RUN,NOT_RUN,NOT_RUN,NOT_RUN
ABL-02,OFF,ON,OFF,NOT_RUN,NOT_RUN,NOT_RUN,NOT_RUN,NOT_RUN,NOT_RUN
ABL-03,OFF,OFF,ON,NOT_RUN,NOT_RUN,NOT_RUN,NOT_RUN,NOT_RUN,NOT_RUN
ABL-04,ON,ON,OFF,NOT_RUN,NOT_RUN,NOT_RUN,NOT_RUN,NOT_RUN,NOT_RUN
ABL-05,ON,OFF,ON,NOT_RUN,NOT_RUN,NOT_RUN,NOT_RUN,NOT_RUN,NOT_RUN
ABL-06,OFF,ON,ON,NOT_RUN,NOT_RUN,NOT_RUN,NOT_RUN,NOT_RUN,NOT_RUN
ABL-07,ON,ON,ON,NOT_RUN,NOT_RUN,NOT_RUN,NOT_RUN,NOT_RUN,NOT_RUN
```
*(No placeholder numerical scores are used prior to actual execution).*

---

## 6. Execution Safety & Standby Status

- **EXP-MLUA-001**: Running undisturbed in background.
- **EXP-MLUA-002**: Prepared and on STANDBY.
- **ABL-00 through ABL-07**: Fully prepared, verified, and placed on **STANDBY**.
- **Execution Rule**: No ablation training shall commence without explicit user instruction.
