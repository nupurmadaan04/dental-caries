# Completed vs. Remaining Work: Capstone Project Status

This document explicitly separates what has been fully completed and empirically verified in the repository from what remains as ongoing or future research work.

---

## 1. Completed Work (Fully Implemented & Verified)

### A. Dataset Curation & Preprocessing Pipeline
- [x] DC1000 dataset integration: 1,000 full-mouth panoramic radiographs structured into 900 training OPGs and 100 sealed test OPGs.
- [x] Aspect-preserving standardized canvas resizing to $768 \times 1536$ pixels (1:2 anatomical aspect ratio).
- [x] CLAHE (Contrast Limited Adaptive Histogram Equalization) contrast normalization.
- [x] Sliding-window patch decomposition into $384 \times 384$ patches at $192\text{-pixel}$ stride (21 patches per panoramic image).
- [x] Partitioning into 20% labeled (530 patches) and 80% unlabeled (1,859 patches) cohorts.

### B. Baseline & Supervised Experiments
- [x] Supervised baseline trials: EXP001, EXP001_EXT, EXP002 (Focal Loss), EXP003 (FPN), EXP004 (Full OPG), EXP005 (DoubleU-Net), and EXP006 (Peak Supervised at 48.39% Dice).
- [x] Empirical verification that supervised learning plateaus below 50% Dice on the small labeled cohort.

### C. Semi-Supervised MLUA Architecture & Training
- [x] ResNet-34 encoder + Feature Pyramid Network (FPN) decoder architecture (~21.28M parameters).
- [x] 4 multi-resolution auxiliary deep supervision heads ($\alpha = [0.1, 0.2, 0.3, 0.4]$).
- [x] Teacher-Student Mean Teacher framework with Exponential Moving Average updates ($\theta = 0.99$).
- [x] Monte Carlo epistemic uncertainty gating ($T=8$ stochastic passes with active dropout $p=0.20$ and dynamic certainty thresholding).

### D. Forensic Investigation & Engineering Remediation
- [x] Isolated the root cause of the EXP-MLUA-002 catastrophic failure (NaN loss at Step 1257) to unsynchronized Teacher BatchNorm running buffers.
- [x] Implemented dual parameter and floating-point BatchNorm buffer EMA synchronization in `src/mlua/engine/`.
- [x] Verified 100% numerical stability with zero NaNs across 10,560 optimization steps (80 epochs) in EXP-MLUA-003.

### E. Model Benchmarking & Sealed Test Evaluation
- [x] Canonical peak checkpoint identified at Epoch 75 (`EXP-MLUA-003_E75_BEST.pth`), achieving **71.867% Validation Dice** (exceeding published literature 71.12%).
- [x] Evaluated strictly once on the 100-case sealed test set on full panoramic images at frozen $\tau = 0.50$: **50.147% Macro Dice**, **52.924% Micro Dice**, **99.872% Specificity**, and **0.0% Zero-Prediction Ratio**.
- [x] Threshold sensitivity sweep across $\tau \in [0.05, 0.95]$ establishing $\tau = 0.50$ as optimal operating balance.

### F. Full Panoramic Reconstruction Engine
- [x] 21-patch sliding-window stitching with 2D Gaussian kernel spatial probability blending to eliminate seam artifacts.

### G. Application & Clinical Decision Support System
- [x] Production FastAPI backend with health checks, analysis endpoints, and conversational routing.
- [x] React 18 + TypeScript clinical review interface with dark/light themes, lesion overlay viewer, and case history.
- [x] 4-tier rule-based severity staging (Normal, Level 1, Level 2, Level 3) grounded in pixel area anchors from the base paper.
- [x] Context-aware Gemini clinical assistant with zero-PHI privacy protection and 3-layer NLU (emotion, intent, generation).
- [x] Automated vector PDF and JSON clinical report generation.
- [x] Test suite: 65 automated tests passing 100% in `backend/test_chat_api.py`.

---

## 2. Remaining Work (Future Research Scope)

### A. Fine-Grained Image-Level Error Analysis
- [ ] Systematic categorization of False Positives across specific anatomical structures (e.g., differentiating cervical burnout vs. interproximal overlap vs. restoration borders).
- [ ] Systematic categorization of False Negatives (e.g., incipient shallow enamel lesions vs. root caries).

### B. Challenging Case & Out-of-Distribution Profiling
- [ ] Evaluating performance specifically on pediatric mixed-dentition OPGs and severe edentulous cases.
- [ ] Profiling model resilience against heavy metal streak artifacts from multiple orthodontic brackets or metallic crowns.

### C. Multi-Threshold Ensemble Exploration
- [ ] Investigating adaptive lesion-specific thresholding (e.g., lower threshold for shallow enamel lesions, higher threshold for deep lesions) to reduce the validation-to-test generalization gap.

### D. Publication & Thesis Deliverables
- [ ] Final academic manuscript drafting for conference/journal submission.
- [ ] Comprehensive literature review expansion comparing with recent 2024–2025 dental AI publications.
- [ ] Preparation of supplementary materials, high-resolution figures, and reproducible code release repository.
