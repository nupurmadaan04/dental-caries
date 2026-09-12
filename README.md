# MLUA: Dental Caries Segmentation from Panoramic X-rays

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Frozen-success.svg)]()
[![Model](https://img.shields.io/badge/Model-ResNet34%20%2B%20FPN%20MLUA-blueviolet.svg)]()

> **Research Benchmark & AI-Assisted Dental Caries Segmentation System**  
> An implementation of Multi-Level Uncertainty-Aware (MLUA) semi-supervised learning for pixel-level dental caries segmentation on panoramic dental radiographs.

---

## Overview

Dental caries (tooth decay) is one of the most prevalent chronic conditions worldwide. Early, precise localization of carious lesions on orthopantomograms (panoramic dental X-rays) is vital for preventative dentistry and treatment planning.

This repository provides a semi-supervised deep learning system that performs **pixel-level binary dental caries segmentation** from panoramic radiographs:

- **Input:** Full-scale panoramic dental radiograph ($768 \times 1536$ standard panoramic canvas).
- **Inference Mechanism:** 21-patch overlapping sliding-window inference ($384 \times 384$ patches at stride $192$) with spatial probability reconstruction.
- **Output:** 
  1. Continuous pixel-level caries probability map ($\in [0.0, 1.0]$).
  2. Binary caries segmentation mask thresholded at the validation-selected operating point $\tau = 0.50$ ($0 = \text{background / sound tooth structure}, 1 = \text{predicted caries region}$).
  3. Diagnostic visualization overlay highlighting predicted lesions directly over the radiograph.

*Note: The model performs binary caries segmentation. It does not perform multi-class severity classification (shallow/middle/deep); lesion severity differentiation requires a separate downstream classification model.*

---

## Problem Statement

Automated segmentation of dental caries on panoramic radiographs presents significant technical challenges:
1. **Severe Class Imbalance:** Caries lesions occupy less than $1\%$ of the total pixels on a panoramic radiograph ($>99\%$ background/non-caries tissue).
2. **Subtle & Diffuse Radiographic Margins:** Incipient and demineralized enamel/dentin lesions exhibit low contrast boundaries against adjacent restorative materials and healthy alveolar bone.
3. **Panoramic Scale vs. Patch Resolution:** Full panoramic images are wide ($1536+$ px) with complex anatomical arch geometry, requiring overlapping localized patch analysis and seamless spatial probability reconstruction.
4. **Generalization to Unseen Patient Radiographs:** Variable patient positioning, bone mineralization densities, dental restorations, and multi-vendor scanner characteristics introduce significant domain shift between training cohorts and external test populations.

---

## Dataset: DC1000

The system is trained and evaluated on the **DC1000** dental caries dataset:
- **Total Panoramic Radiographs:** 1,000 raw panoramic images.
- **Annotated Cohort:** 493 fully cleaned and expert-verified images with $>7,500$ annotated caries lesions.
- **Severity Annotations (Dataset Ground Truth):** Classified by clinicians into shallow, medium, and deep caries.
- **Training Patch Repository:** 2,389 extracted $384 \times 384$ patches (530 labeled patches at $20\%$ active supervision rate; 1,859 unlabeled patches for consistency regularization).
- **Independent Sealed Test Set:** Exactly 100 panoramic test cases ($768 \times 1536$ resolution with paired ground truth binary masks) strictly isolated and sealed from training, checkpoint selection, and threshold tuning.

---

## Methodology

```
                                    +-----------------------------------------+
                                    |     Unlabeled Training Patches          |
                                    +-----------------------------------------+
                                                         |
                                                         v
+-----------------------+     Teacher EMA Buffer & Param |
|   Labeled Patches     |     Synchronization (theta=0.99)|
+-----------------------+                                v
           |                         +----------------------------------------+
           v                         |             Teacher Model              |
+-----------------------+            |       (ResNet-34 + FPN Decoder)        |
|     Student Model     |            +----------------------------------------+
| (ResNet-34 + FPN MLUA)|                                | (Monte Carlo Perturbation)
+-----------------------+                                v
           |                         +----------------------------------------+
           |                         |   Multi-Level Uncertainty Map (U)      |
           |                         +----------------------------------------+
           |                                             |
           v                                             v
+-----------------------------------------------------------------------------+
| Supervised Loss (Dice + CE)  +  Uncertainty-Weighted Consistency Loss       |
+-----------------------------------------------------------------------------+
```

### Key Architectural Principles:
1. **Backbone & Decoder:** ResNet-34 encoder with a 4-stage Feature Pyramid Network (FPN) decoder generating deep supervision auxiliary outputs at stages 1–4 and a fused segmentation head.
2. **Semi-Supervised Consistency Learning:** Teacher-Student architecture enforcing prediction consistency on unlabeled patches under perturbations.
3. **Multi-Level Uncertainty Awareness (MLUA):** Monte Carlo sampling estimates pixel-wise epistemic uncertainty, adaptively down-weighting ambiguous lesion boundaries during consistency regularization.
4. **Teacher BatchNorm Buffer Synchronization:** Both model parameters and BatchNorm running statistics (`running_mean`, `running_var`) are synchronized via exponential moving average ($\theta = 0.99$), eliminating teacher-student feature drift and ensuring numerical stability.

---

## Experimental Progression

| Experiment | Configuration & Purpose | Key Findings / Outcome |
| :--- | :--- | :--- |
| **EXP-MLUA-001** | Initial FP16 baseline & compute profiling | Validated 21-patch sliding window pipeline; revealed numerical sensitivity under mixed precision. |
| **EXP-MLUA-002** | Initial 50-epoch semi-supervised training | Reached Epoch 9; encountered numerical instability at global step 1,257 due to unsynchronized teacher BatchNorm buffers. |
| **EXP-MLUA-003** | Controlled remediation with full EMA buffer sync | **Completed 60/60 epochs (7,920 steps) with 100% numerical stability (0 NaNs, 0 Infs). Established peak validation performance.** |

---

## Final Model Specification

- **Architecture:** ResNet-34 + FPN MLUA (Auxiliary Deep Supervision + Fused Head)
- **Selected Checkpoint:** [`outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E56_FINAL.pth`](file:///c:/Users/devin/MLUA/outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E56_FINAL.pth)
- **Epoch:** `56`
- **Global Step:** `7392`
- **Operating Threshold:** $\tau = 0.50$
- **Selection Criterion:** Highest observed validation Dice score within the EXP-MLUA-003 training run.

---

## Validation Results

Evaluated on the canonical validation cohort using the standard 21-patch sliding window protocol at $\tau = 0.50$:

| Metric | Validation Performance |
| :--- | :---: |
| **Validation Dice** | **`65.623%`** |
| **Validation IoU / Jaccard** | **`49.854%`** |
| **Validation Precision** | **`69.009%`** |
| **Validation Recall (Sensitivity)** | **`63.649%`** |
| **Validation Specificity** | **`99.753%`** |
| **Validation Loss** | **`0.7639`** |

### Threshold Selection
A systematic, read-only sensitivity sweep across 19 candidate thresholds ($\tau \in [0.05, 0.95]$ with step $0.05$) was conducted on the validation set. **$\tau = 0.50$ produced the highest observed validation Dice among the 19 evaluated thresholds.** The operating threshold was consequently frozen at $\tau = 0.50$ for all downstream evaluations.

---

## Independent Sealed Test Results

The frozen model was evaluated in a single-pass, read-only benchmark on the **100-case sealed test set** (`dataset/test/`):

| Metric | Canonical Validation (E56, $\tau=0.50$) | Sealed Test Macro (Case Mean) | Sealed Test Micro (Pixel Sum) | Absolute Difference ($\Delta = \text{Test}_{\text{macro}} - \text{Val}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Dice Coefficient** | `65.623%` | **`43.041%`** | **`43.391%`** | `-22.582%` |
| **IoU / Jaccard** | `49.854%` | **`29.057%`** | **`27.707%`** | `-20.797%` |
| **Precision** | `69.009%` | **`41.244%`** | **`37.795%`** | `-27.765%` |
| **Recall (Sensitivity)** | `63.649%` | **`52.896%`** | **`50.931%`** | `-10.753%` |
| **Specificity** | `99.753%` | **`99.630%`** | **`99.630%`** | `-0.123%` |
| **Zero-Prediction Ratio** | `0.0%` | **`0.0%`** (0 / 100 cases) | — | `0.0%` |

- **Total Test Pixels Evaluated:** $117,964,800$ pixels ($100 \text{ cases} \times 768 \times 1536$)
- **Pixel Confusion Totals:** $\text{TP} = 263,935$, $\text{FP} = 434,392$, $\text{FN} = 254,282$, $\text{TN} = 117,012,191$.
- **Generalization Gap Analysis:** The $-22.58\%$ absolute difference in Dice reflects domain variance on unseen test radiographs, primarily manifested as lower precision ($41.24\%$ vs. $69.01\%$) due to false positive detections on subtle anatomical structures, while specificity ($99.63\%$) and background discrimination remained exceptionally high.

---

## Inference Pipeline

```
+-------------------------------------------------------------+
|                 Full Panoramic Dental X-ray                 |
|                     (e.g., 768 x 1536)                      |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                        Preprocessing                        |
|       (Grayscale .convert("L"), Float32 / 255.0 Norm)       |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                   384x384 Patch Extraction                  |
|     (21 Overlapping Patches, Stride 192, 3x7 Grid)          |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                 MLUA E56 Model Forward Pass                 |
|                   (torch.inference_mode())                  |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|            Spatial Overlap Probability Reconstruction       |
|            (Normalizing by overlapping patch counts)         |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|             Sigmoid Probability Map ([0.0, 1.0])             |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                Operating Threshold (tau = 0.50)             |
+-------------------------------------------------------------+
                              |
               +--------------+--------------+
               |                             |
               v                             v
+-----------------------------+ +-----------------------------+
|    Binary Mask ({0, 1})     | |  Alpha Blended RGB Overlay  |
|  (Predicted Caries Regions) | | (Visual Diagnostic Preview) |
+-----------------------------+ +-----------------------------+
```

---

## Integration Preflight Verification

An end-to-end integration preflight sanity check was executed on a representative non-test training radiograph (`101.png`):
- **Model Checkpoint:** `EXP-MLUA-003_E56_FINAL.pth` loaded cleanly (100% strict key matching).
- **Parameter Health:** 0 NaNs, 0 Infs across all model weights.
- **Inference Verification:** 21 patches processed; reconstructed to $768 \times 1536$ continuous map ($\min = 0.000000, \max = 0.998185, \text{mean} = 0.015858$).
- **Mask Generation:** $\tau = 0.50$ produced 7,271 positive pixels ($0.616\%$ area coverage); zero prediction = `False`.
- **Numerical Safety:** 0 NaNs, 0 Infs throughout the entire inference pipeline.
- **Diagnostic Preflight Status:** **PASS** (Artifacts archived in `outputs/diagnostics/FINAL_INTEGRATION_PREFLIGHT/`).

---

## Repository Structure

```text
MLUA/
├── configs/                                  # Configuration files
│   ├── evaluation/
│   │   └── sealed_test_100_config.yaml       # Sealed test evaluation configuration
│   ├── experiments/
│   │   ├── exp_mlua_002_historical_config.yaml
│   │   ├── exp_mlua_003_final_config.yaml    # EXP-MLUA-003 canonical training config
│   │   ├── ablation/                         # Ablation configs (ABL-00 to ABL-07)
│   │   └── mc_sampling/                      # Monte Carlo configs (MC-05 to MC-160)
│   └── mlua_default_config.yaml              # Default configuration
│
├── data/                                     # Raw dataset storage
│   └── raw/
│       └── DC1000_dataset/
│           ├── org_train_dataset/            # Original training panoramic dental X-rays
│           ├── org_test_dataset/             # Original test set records
│           └── train/                        # 384x384 training patches & labels
│
├── dataset/                                  # Prepared evaluation datasets
│   └── test/                                 # 100 Sealed Panoramic Test Cases (768x1536)
│       ├── colors/ & colors_cut/
│       ├── images/ & images_cut/
│       └── labels/ & labels_cut/
│
├── docs/                                     # Structured research and engineering documentation
│   ├── research/
│   │   ├── experiment_history/               # Experiment logs & pretraining audits
│   │   ├── forensic_audits/                  # Historical root-cause & numerical audits
│   │   ├── threshold_analysis/               # Threshold sweep reports
│   │   └── final_evaluation/                 # Evaluation traceability reports
│   └── engineering/
│       └── integration/                      # Frontend mapping & optimization reports
│
├── evaluate/                                 # Evaluation utility routines
│   ├── common.py
│   └── evaluation_utils.py                   # Data loaders & image transformations
│
├── frontend/                                 # React + Vite + Tailwind Web Application
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── services/
│   ├── package.json
│   └── vite.config.ts
│
├── outputs/                                  # Experiment outputs & checkpoints
│   ├── diagnostics/
│   │   ├── EXP-MLUA-003_SEALED_TEST/         # Final 100-case test report & per-case CSV
│   │   ├── EXP-MLUA-003_VALIDATION_THRESHOLD_ANALYSIS/ # Validation threshold sweep
│   │   ├── EXP-MLUA-003_E60_TRAINING_EXTENSION_AUDIT/  # E51-E60 extension audit
│   │   ├── EXP-MLUA-003_E50_MILESTONE_AUDIT/           # E50 audit
│   │   └── FINAL_INTEGRATION_PREFLIGHT/      # Preflight overlays & report
│   └── experiments/
│       ├── EXP-MLUA-001_HISTORICAL/
│       ├── EXP-MLUA-002_HISTORICAL/
│       └── EXP-MLUA-003_FINAL/               # Final research experiment directory
│           ├── checkpoints/
│           │   ├── EXP-MLUA-003_E56_FINAL.pth # Final selected checkpoint (E56 / Step 7392)
│           │   └── EXP-MLUA-003_E60_LATEST.pth
│           ├── EXP-MLUA-003_FULL_TRAINING_HISTORY.csv
│           ├── FINAL_MODEL_SELECTION.md
│           ├── FINAL_RESEARCH_FREEZE.md
│           └── TRAINING_COMPLETION_STATUS.md
│
├── research_archive/                         # Archived research scripts and experimental code
│   └── scripts/
│
├── src/                                      # Core Python MLUA Package
│   └── mlua/
│       ├── data/                             # Dataset loaders & samplers
│       ├── engine/                           # Training engines & optimizers
│       ├── evaluation/                       # Panoramic inference & reconstruction
│       └── models/
│           └── fpn.py                        # ResNet-34 + FPN MLUA architecture
│
├── requirements.txt                          # Python dependencies
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-compatible GPU (optional, CPU execution supported)
- Node.js 18+ (for frontend dashboard)

### Setup Python Environment

```bash
# Clone repository
git clone https://github.com/Zzz512/MLUA.git
cd MLUA

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Python Inference on a Panoramic Radiograph

```python
import torch
import numpy as np
from PIL import Image
from src.mlua.models.fpn import Net
from src.mlua.evaluation.final_100_evaluation import (
    extract_panoramic_patches,
    reconstruct_panoramic,
)

# 1. Load Model and Frozen Checkpoint
checkpoint_path = "outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E56_FINAL.pth"
model = Net(in_c=1, out_c=1, encoder_name="resnet34", encoder_weights=None)
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_stu_state_dict"])
model.eval()

# 2. Load and Preprocess Panoramic Image (768 x 1536)
raw_img = Image.open("data/raw/DC1000_dataset/org_train_dataset/images/101.png").convert("L")
panoramic_img = raw_img.resize((1536, 768), Image.Resampling.BILINEAR)
img_np = np.asarray(panoramic_img)

# 3. Extract 21 overlapping 384x384 patches (stride 192)
patches, coords = extract_panoramic_patches(img_np, patch_size=384, stride=192)
patch_tensors = [torch.from_numpy(p.astype(np.float32) / 255.0).unsqueeze(0) for p in patches]
batch_tensor = torch.stack(patch_tensors, dim=0) # [21, 1, 384, 384]

# 4. Model Forward Pass
with torch.inference_mode():
    fused_logits, _ = model(batch_tensor)
    pred_probs = torch.sigmoid(fused_logits).squeeze(1).cpu().numpy()

# 5. Reconstruct Panoramic Probability Map & Apply tau = 0.50 Threshold
prob_map = reconstruct_panoramic(list(pred_probs), coords, full_shape=(768, 1536), patch_size=384)
binary_mask = (prob_map >= 0.50).astype(np.uint8)

print(f"Inference complete. Detected caries positive pixels: {binary_mask.sum()}")
```

### 2. Frontend Visualization Dashboard

```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Start local development server
npm run dev
```

*Note: Direct backend REST API integration connecting the frontend to the Python model server is the next engineering phase.*

---

## Research Limitations

1. **Generalization Gap:** An observed $-22.58\%$ absolute difference in Dice exists between the canonical validation set ($65.62\%$) and the sealed test set ($43.04\%$), reflecting radiographic domain shift on unseen patient populations.
2. **Precision Reduction on Out-of-Distribution Cases:** Macro Precision on the test cohort ($41.24\%$) indicates false positive sensitivity on challenging anatomical structures and low-contrast dental margins.
3. **Case-Level Variance:** Significant inter-case variance across test radiographs (standard deviation of $17.87\%$ in Dice) indicates sensitivity to patient positioning and bone density variations.
4. **Binary Task Scope:** The model segments general caries regions and does not differentiate between lesion depth stages (shallow, medium, deep).
5. **Research Benchmark Nature:** This repository provides an academic research benchmark and does not represent a certified medical diagnostic tool.

---

## Future Work

- **Domain Generalization:** Integrating adaptive test-time augmentation (TTA) and domain-adversarial regularization to close the generalization gap.
- **False-Positive Reduction:** Implementing multi-scale boundary refinement and morphological post-processing filters.
- **Multi-Class Severity Classification:** Coupling the segmentation backbone with a hierarchical classifier to distinguish shallow, middle, and deep caries.
- **Multi-Center External Validation:** Testing on external radiograph datasets across diverse multi-vendor panoramic imaging systems.
- **Full Backend API Deployment:** Building a high-throughput FastAPI microservice to connect the Python inference engine to the interactive frontend dashboard.

---

## Citation & Academic Background

This implementation is grounded in the MLUA semi-supervised framework:

```bibtex
@article{mlua2023,
  title={Multi-level uncertainty aware learning for semi-supervised dental panoramic caries segmentation},
  journal={Neurocomputing},
  volume={540},
  pages={126208},
  year={2023},
  publisher={Elsevier}
}
```

---

## Final Disclaimer

> **This system is intended for research and educational purposes. Predicted regions represent model-generated segmentation outputs and should not be interpreted as a clinical diagnosis or substitute for professional dental assessment.**
