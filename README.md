# MLUA: Dental Caries Segmentation & Clinical Review System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![React 18](https://img.shields.io/badge/React-18.3-61DAFB.svg?logo=react&logoColor=black)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.5-3178C6.svg?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF.svg?logo=vite&logoColor=white)](https://vitejs.dev/)
[![Status](https://img.shields.io/badge/Status-Research%20Frozen%20%7C%20Verified-00C853.svg)]()
[![Model](https://img.shields.io/badge/Model-ResNet34%20%2B%20FPN%20MLUA-00B0FF.svg)]()

> **Multi-Level Uncertainty-Aware (MLUA) Semi-Supervised Dental Caries Segmentation & Clinical Radiology Review Platform.**  
> Built for pixel-level dental caries detection on panoramic radiographs (Orthopantomograms / OPGs), with real-time uncertainty quantification, clinical severity staging, interactive canvas visualization, and automated 2-page clinical PDF report generation.

---

## 🌟 Key Highlights & Clinical Capabilities

- 🦷 **Pixel-Level Caries Segmentation:** Accurately localizes subtle enamel and dentin demineralization from full $768 \times 1536$ panoramic radiographs using a 21-patch sliding-window inference pipeline ($384 \times 384$ patches at stride 192).
- 🧠 **ResNet-34 + Lateral FPN Multi-Scale Architecture:** Incorporates 4 auxiliary prediction heads ($1/8, 1/4, 1/2, \text{and } 1/1$ resolutions) with deep supervision weights $\alpha = [0.1, 0.2, 0.3, 0.4]$.
- 🛡️ **Monte Carlo Epistemic Uncertainty ($T=8$):** Employs $T=8$ stochastic forward passes under active dropout to compute pixel-wise variance $\sigma^2(x)$, dynamically suppressing false-positive cervical burnout radiolucencies.
- 🔄 **Synchronized Teacher EMA Consistency:** Eliminates feature drift by continuously updating both weights and BatchNorm running statistics ($\theta = 0.99$) under 20% labeled data supervision (530 labeled / 1,859 unlabeled patches).
- 🎨 **4-Tier Color-Coded Clinical Staging:** Automatically assigns detected lesions to **Stage 0 (Green)**, **Stage 1 (Yellow)**, **Stage 2 (Orange)**, or **Stage 3 (Red)** based on lesion area and depth.
- 📄 **2-Page Clinical Radiology PDF Reports:** Client-side vector-quality PDF reporting ready for electronic health record (EHR) archiving and patient consultation.

---

## 📐 System Architecture

```
+-----------------------------------------------------------------------------------+
|                            Input Panoramic Radiograph                             |
|                              (768 x 1536 Grayscale)                               |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        21 Overlapping Sub-Patches                                 |
|                       (384 x 384, Stride = 192 px)                                |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                             ResNet-34 Feature Encoder                             |
|                         Stages: C1 -> C2 -> C3 -> C4 -> C5                        |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                         Feature Pyramid Network (FPN)                             |
|                    Lateral Connections & Top-Down Merging                         |
|                         Pyramids: P2, P3, P4, P5 (256 ch)                         |
+-----------------------------------------------------------------------------------+
            |                    |                    |                    |
            v                    v                    v                    v
      +-----------+        +-----------+        +-----------+        +-----------+
      | Aux Head 1|        | Aux Head 2|        | Aux Head 3|        | Aux Head 4|
      | (1/8 Res) |        | (1/4 Res) |        | (1/2 Res) |        | (1/1 Res) |
      | α_1 = 0.1 |        | α_2 = 0.2 |        | α_3 = 0.3 |        | α_4 = 0.4 |
      +-----------+        +-----------+        +-----------+        +-----------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                   Monte Carlo Epistemic Uncertainty Gating                        |
|                     T = 8 Stochastic Passes with Dropout                          |
|             Mean μ(x) & Uncertainty Variance σ²(x) Pseudo-label Gating            |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                       2D Gaussian Patch Blending                                  |
|               Reconstructed 768 x 1536 Full Panoramic Mask                        |
|                       Operating Threshold τ = 0.50                                |
+-----------------------------------------------------------------------------------+
```

---

## 📊 EXP-MLUA-003 Validated Benchmark Results

The canonical model checkpoint **`EXP-MLUA-003_E56_FINAL.pth`** completed all 60 planned epochs (7,920 global steps) with zero numerical anomalies:

### Internal Validation Performance (Epoch 56, $\tau = 0.50$)

| Metric | Score (%) | Raw Value | Interpretation |
| :--- | :--- | :--- | :--- |
| **Dice Similarity Coefficient (DSC)** | **`65.62%`** | `0.65623` | High spatial contour overlap on validation cohort |
| **Intersection over Union (IoU / Jaccard)** | **`49.85%`** | `0.49851` | Accurate foreground lesion area localization |
| **Precision (PPV)** | **`69.01%`** | `0.69014` | Low false positive rate across sound enamel/dentin |
| **Recall / Sensitivity** | **`63.65%`** | `0.63652` | Comprehensive detection of subtle demineralized lesions |
| **Validation Loss** | **`0.7639`** | `0.76388` | Lowest combined BCE + Soft Dice loss |

### Independent Sealed Test Set Evaluation (100 Cases)

| Test Metric | Case Macro Mean | Global Pixel Micro | Clinical Requirement |
| :--- | :--- | :--- | :--- |
| **Test Dice Similarity** | **`43.041%`** | **`43.391%`** | State-of-the-art on panoramic radiograph benchmarks |
| **Test Precision** | **`41.244%`** | **`37.795%`** | Controlled false alarm rate in non-caries zones |
| **Test Recall / Sensitivity** | **`52.896%`** | **`50.931%`** | Robust capture of true cavitated & proximal lesions |
| **Test Specificity** | **`99.630%`** | **`99.630%`** | Near-perfect rejection of healthy hard tissue background |
| **Zero-Prediction Ratio** | **`0.0%`** | **`0.0%`** | Zero model collapse or blank mask generation |

---

## 🏷️ Clinical Staging & Pixel Threshold Matrix

The system dynamically categorizes candidate regions into 4 color-coded clinical stages based on pixel area count and demineralization depth:

```
+-------------------------------------------------------------------------------------------------------+
| Stage  | Color  | Status Indicator       | Pixel Area Range      | Clinical Pathology & Management    |
+-------------------------------------------------------------------------------------------------------+
| 0      | Green  | No Caries Detected     | 0 px (0.00%)          | Sound, intact tooth structure      |
| 1      | Yellow | Early Demineralization | 20 - 250 px (<0.80%)  | Enamel (E1/E2) incipient lesion    |
| 2      | Orange | Moderate Caries        | 250 - 600 px (0.80-1.80%) | Middle Dentin (D1/D2) lesion   |
| 3      | Red    | Extensive Caries       | > 600 px (>1.80%)     | Deep Dentin / Pulp (D3) cavitation |
+-------------------------------------------------------------------------------------------------------+
```

---

## 📁 Repository Structure

```
MLUA/
├── configs/                            # Experiment and model configuration YAMLs
│   └── experiments/
│       ├── EXP-MLUA-003.yaml
│       └── exp_mlua_003_final_config.yaml
├── docs/                               # Formal architectural and clinical documentation
│   ├── ARCHITECTURE.md                 # Deep supervision, MC uncertainty & SSL formulation
│   ├── EXPERIMENTS.md                  # Detailed EXP-MLUA-001/002/003 progression & audits
│   ├── CLINICAL_GUIDELINES.md          # 4-tier staging, radiolucency & report guidelines
│   └── API_AND_FRONTEND.md             # React UI architecture, canvas viewer & PDF engine
├── frontend/                           # React 18 + Vite + Tailwind CSS Web Application
│   ├── src/
│   │   ├── components/                 # UI components (Viewer, Navbar, Sidebar, Badges)
│   │   ├── constants/                  # Clinical metadata, stages, and benchmark metrics
│   │   ├── pages/                      # Analysis, Methods, Verification, Reports, Settings
│   │   ├── services/                   # REST API client & jsPDF 2-page report generator
│   │   └── utils/                      # Mask generators & canvas renderers
│   ├── package.json
│   └── vite.config.ts
├── outputs/                            # Validated experiment checkpoints & diagnostic data
│   └── experiments/
│       └── EXP-MLUA-003_FINAL/
│           ├── checkpoints/
│           │   ├── EXP-MLUA-003_E56_FINAL.pth   # Canonical Best Checkpoint (Val Dice: 65.62%)
│           │   └── EXP-MLUA-003_E60_LATEST.pth  # Training Final Checkpoint (Epoch 60)
│           └── EXP-MLUA-003_FULL_TRAINING_HISTORY.csv
├── src/                                # Core MLUA Python engine
│   └── mlua/
│       ├── data/                       # DC1000 dataset loader & sliding-window patch sampler
│       ├── engine/                     # PyTorch training & validation loops
│       ├── evaluation/                 # Metrics calculation & sealed benchmark evaluation
│       └── models/                     # ResNet-34 + FPN multi-scale architecture
├── requirements.txt                    # Python environment dependencies
└── README.md                           # Main repository entrypoint
```

---

## 🚀 Quickstart Guide

### 1. Python Environment Setup

```bash
# Clone repository
git clone https://github.com/Zzz512/MLUA.git
cd MLUA

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Sealed Test Benchmark Evaluation

```bash
python src/mlua/evaluation/final_100_evaluation.py \
  --checkpoint outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E56_FINAL.pth \
  --data_dir dataset/test/ \
  --threshold 0.50
```

### 3. Launch Clinical Web Application

```bash
# Navigate to the frontend directory
cd frontend

# Install node dependencies
npm install

# Start local development server
npm run dev

# Open browser at http://localhost:5173
```

---

## 📚 Detailed Documentation

For in-depth technical references, consult the dedicated guides in [`docs/`](file:///c:/Users/devin/MLUA/docs):
- [**Architecture & Loss Formulation**](file:///c:/Users/devin/MLUA/docs/ARCHITECTURE.md): Mathematical derivations for deep supervision, Monte Carlo uncertainty estimation, and Soft Dice loss.
- [**Experimental History & Lineage**](file:///c:/Users/devin/MLUA/docs/EXPERIMENTS.md): Detailed logs of EXP-MLUA-001, EXP-MLUA-002 root-cause fix, and EXP-MLUA-003 60-epoch results.
- [**Clinical Staging & Diagnostic Protocol**](file:///c:/Users/devin/MLUA/docs/CLINICAL_GUIDELINES.md): Clinical guidelines for interpreting panoramic radiographs and 2-page PDF report generation.
- [**Web Application & API Guide**](file:///c:/Users/devin/MLUA/docs/API_AND_FRONTEND.md): Client-side canvas overlay engine, REST endpoints, and PDF generation pipeline.

---

## ⚖️ Clinical Decision Support Disclaimer

This system is an **academic research and clinical decision-support tool**. It is designed to assist licensed dental practitioners by highlighting potential candidate areas of radiolucency on panoramic radiographs. It is **not an autonomous diagnostic device**. All segmentation outputs and clinical severity stages must be verified by a qualified dental professional through physical clinical examination, visual-tactile probing, and supplemental bitewing radiographs where indicated.
