# EXP-MLUA-003 Final Model Freeze

### Experiment:
EXP-MLUA-003

### Final Selected Checkpoint:
`outputs/experiments/EXP-MLUA-003/checkpoints/EXP-MLUA-003_BEST.pth`

### Selected Epoch:
56

### Global Step:
7392

### Selection Criterion:
Highest observed validation Dice within EXP-MLUA-003.

### Validation Metrics (at τ = 0.50):
- **Validation Dice:** 65.623%
- **Validation IoU:** 49.854%
- **Validation Precision:** 69.009%
- **Validation Recall:** 63.649%
- **Validation Specificity:** 99.753%
- **Validation Loss:** 0.7639

### Operating Threshold:
τ = 0.50

### Important Threshold Note:
τ = 0.50 produced the highest observed validation Dice among the 19 evaluated validation thresholds ($\tau \in [0.05, 0.95]$) during the post-training sensitivity analysis.

### Scientific Designation & Scope:
- **Designation:** "final selected EXP-MLUA-003 checkpoint" or "frozen EXP-MLUA-003 E56 checkpoint".
- **Non-Clinical Clarification:** This is a research benchmark and must NOT be described as a "globally best model", "clinically validated model", or "clinically accurate model".
- **Freeze Status:** The model weights and architecture are strictly frozen for downstream integration.

### Lifecycle Milestone Record:
- **Training Status:** Complete (60/60 epochs, 7,920 global steps).
- **Extension Status:** E51–E60 controlled extension complete.
- **Checkpoint Selection:** E56 is the final selected checkpoint.
- **Threshold Selection:** Complete ($\tau = 0.50$ frozen).
- **Sealed Test Set Evaluation:** Complete (100 sealed cases evaluated once; Macro Dice: 43.041%, Micro Dice: 43.391%).
- **Optimization Ban:** No further test-driven optimization, threshold tuning, or weight modification is permitted.
