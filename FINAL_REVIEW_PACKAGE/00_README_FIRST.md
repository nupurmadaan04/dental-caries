# 00_README_FIRST: Capstone Project Mentor Review Guide

## Project Title
**Multi-Level Uncertainty-Aware (MLUA) Dental Caries Clinical AI**  
*Semi-Supervised Semantic Segmentation on Dental Panoramic Radiographs (OPGs)*

---

## Key Model & Metric Summary

- **Canonical Active Model:** `EXP-MLUA-003_E75_BEST.pth`
- **Selected Epoch:** Epoch 75 (Global Step 9900; Concluded at Epoch 80 / Step 10,560)
- **Model Architecture:** ResNet-34 Encoder + Feature Pyramid Network (FPN) Decoder (~21.28M parameters)
- **Operating Decision Threshold:** $\tau = 0.50$ (Fixed prior to evaluation, zero test-time tuning)
- **Best Validation Dice:** **71.867%** (Exceeds published literature benchmark of 71.12% by +0.747 pp)
- **Validation IoU / Precision / Recall:** IoU: 57.35% | Precision: 78.13% | Recall: 67.34% | Specificity: 99.82%
- **Final Sealed-Test Macro Dice:** **50.147%** (Across 100 independent full-mouth radiographs)
- **Final Sealed-Test Micro Dice:** **52.924%** (Across 117,964,800 evaluated test pixels)
- **Sealed-Test Specificity:** **99.872%** (Zero-prediction collapse ratio: 0.0%)

---

## Main Research Story

```text
Supervised Baselines (EXP001-EXP006, ceiling <48.4% Dice due to 530 labeled patches)
  ↓
Semi-Supervised MLUA Paradigm (Leveraging 1,859 unannotated clinical patches)
  ↓
Catastrophic Numerical Failure (EXP-MLUA-002 loss exploded to NaN at Step 1257)
  ↓
Forensic Hook Investigation (Discovered exponential 10^18 activation scaling in Teacher)
  ↓
Root Cause Discovery (Teacher BatchNorm running buffers were omitted from PyTorch EMA loop)
  ↓
Dual Buffer Synchronization Fix (Synchronized weights AND running_mean / running_var)
  ↓
Finite Convergence in EXP-MLUA-003 (10,560 steps / 80 epochs with ZERO NaNs)
  ↓
Epoch 75 Global Optimum (71.87% Validation Dice, beating 71.12% literature benchmark)
  ↓
Final Sealed-Test Evaluation (Macro Dice 50.15%, Specificity 99.87% on 100 full cases)
  ↓
Clinical Review System Integration (FastAPI + React 18 + Zero-PHI Conversational Assistant)
```

---

## Recommended Review Navigation Order for Mentor

If you have limited time, we recommend reviewing this package in the following order:

1. **`01_PROGRESS_REPORT/`**  
   Read [`FINAL_PROGRESS_REPORT.pdf`](file:///01_PROGRESS_REPORT/FINAL_PROGRESS_REPORT.pdf) or [`FINAL_PROGRESS_REPORT.md`](file:///01_PROGRESS_REPORT/FINAL_PROGRESS_REPORT.md) for the complete 12-section project progress narrative.

2. **`03_EXPERIMENTS/`**  
   - [`EXPERIMENT_PROGRESS.md`](file:///03_EXPERIMENTS/EXPERIMENT_PROGRESS.md): Master table of all supervised and MLUA trials.  
   - [`FAILED_EXPERIMENTS_AND_FIXES.md`](file:///03_EXPERIMENTS/FAILED_EXPERIMENTS_AND_FIXES.md): The forensic audit of the Step 1257 failure and the buffer synchronization fix.  
   - [`E75_FINAL_EXPERIMENT.md`](file:///03_EXPERIMENTS/E75_FINAL_EXPERIMENT.md): Full technical breakdown of our canonical active model.

3. **`06_SCREENSHOTS/`**  
   Inspect [`SCREENSHOT_INDEX.md`](file:///06_SCREENSHOTS/SCREENSHOT_INDEX.md) and the 14 visual screenshots covering the live UI, segmentation overlays, loss curves, and uncertainty maps.

4. **`07_TERMINAL_EVIDENCE/`**  
   Review [`TRAINING_LOG_EVIDENCE.txt`](file:///07_TERMINAL_EVIDENCE/TRAINING_LOG_EVIDENCE.txt) and the 4 rendered terminal screenshots showing exact console training outputs.

5. **`02_DATASET/`**  
   Inspect [`DATASET_README.md`](file:///02_DATASET/DATASET_README.md), [`DATASET_ANALYSIS.md`](file:///02_DATASET/DATASET_ANALYSIS.md), and the verified dataset archive `DC1000_dataset.zip`.

6. **`08_MENTOR_NOTES/`**  
   Review [`MENTOR_EXPLANATION_NOTES.md`](file:///08_MENTOR_NOTES/MENTOR_EXPLANATION_NOTES.md) for short, direct answers to common viva questions.

---
*All files in this review package are derived strictly from empirical logs, source code, and verified checkpoints present in repository `nupurmadaan04/dental-caries`.*
