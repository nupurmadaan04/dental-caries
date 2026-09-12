# Research Freeze

### Status:
**FROZEN**

### Final Experiment:
EXP-MLUA-003 (ResNet-34 + FPN MLUA Semi-Supervised Segmentation)

### Final Selected Checkpoint:
`outputs/experiments/EXP-MLUA-003/checkpoints/EXP-MLUA-003_BEST.pth` (Epoch 56 / Global Step 7392)

### Training:
60 epochs / 7,920 global steps (Native FP32, Teacher EMA buffer & parameter synchronization verified, 0 NaNs, 0 Infs).

### Selected Threshold:
τ = 0.50 (Validated as globally optimal for validation Dice across 19 candidate operating thresholds).

### Validation Selection:
Epoch 56 selected strictly by canonical validation performance prior to test set access:
- Validation Dice: **65.623%**
- Validation IoU: **49.854%**
- Validation Precision: **69.009%**
- Validation Recall: **63.649%**
- Validation Specificity: **99.753%**

### Independent Test:
100-case sealed test evaluated once after model and threshold were frozen:
- Macro Test Dice: **43.041%** (Macro IoU: 29.057%, Precision: 41.244%, Recall: 52.896%, Specificity: 99.630%)
- Micro Test Dice: **43.391%** (Micro IoU: 27.707%, Precision: 37.795%, Recall: 50.931%, Specificity: 99.630%)
- Zero-Prediction Ratio: **0.0%**

### Test Optimization:
**None.** No threshold sweeping, post-processing tuning, or model alteration was performed on the sealed test set.

### Test Leakage:
**Zero test leakage.** No test data was used for training, checkpoint selection, threshold selection, or test-time optimization.

---

### Core Research Freeze Directive:
> "Research artifacts are frozen. Any future changes must be implemented as a separate engineering/integration change and must not modify the frozen research results."

---

### Scientific Scope & Terminology:
The final test evaluation represents an independent scientific research benchmark for panoramic dental radiograph segmentation and must **not** be described as "clinical validation", "clinical accuracy", or "production clinical performance".

---

### Verified Frozen Artifact Registry:
The following research directories and historical checkpoints have been verified intact and frozen:

1. `outputs/experiments/EXP-MLUA-003/checkpoints/EXP-MLUA-003_BEST.pth` [Verified: E56, Step 7392, Finite]
2. `outputs/experiments/EXP-MLUA-003/checkpoints/EXP-MLUA-003_LATEST.pth` [Verified: E60, Step 7920, Finite]
3. `outputs/diagnostics/EXP-MLUA-003_THRESHOLD_ANALYSIS/` [Verified: 19-threshold sweep reports]
4. `outputs/diagnostics/EXP-MLUA-003_FINAL_TEST/` [Verified: 100-case test report & per-case CSV]
5. `outputs/diagnostics/EXP-MLUA-003_E60_EXTENSION_AUDIT/` [Verified: E51–E60 audit report]
6. `outputs/diagnostics/EXP-MLUA-003_E50_MILESTONE_AUDIT/` [Verified: E1–E50 milestone audit]
7. `outputs/diagnostics/EXP-MLUA-003_E20_MILESTONE_AUDIT/` [Verified: E20 milestone audit]
8. `outputs/diagnostics/EXP-MLUA-002_E10_FORENSIC/` [Verified: Historical crash root-cause investigation]
9. `outputs/diagnostics/EXP-MLUA-002_E10_PRECISION_AUDIT/` [Verified: Precision & numerical instability audit]
10. `outputs/diagnostics/EXP-MLUA-002_E10_ACTIVATION_FORENSIC/` [Verified: Activation explosion diagnosis]
11. `outputs/diagnostics/EXP-MLUA-002_E10_EMA_BUFFER_AUDIT/` [Verified: Teacher BN buffer synchronization forensic]
12. `outputs/experiments/EXP-MLUA-002/checkpoints/EXP-MLUA-002_BEST.pth` [Verified: Untouched historical baseline]
13. `outputs/experiments/EXP-MLUA-002/checkpoints/EXP-MLUA-002_LATEST.pth` [Verified: Untouched historical baseline]
