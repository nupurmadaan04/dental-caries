# EXP-MLUA-001 E19-E22 Diagnostic Artifacts Directory

This directory contains forensic audit artifacts investigating the validation Dice decline in `EXP-MLUA-001` between Epoch 19 (14.705%) and Epoch 22 (0.235%).

## Contents
- [`FORENSIC_AUDIT_REPORT.md`](file:///c:/Users/devin/MLUA/outputs/experiments/EXP-MLUA-001/diagnostics/e19_e22/FORENSIC_AUDIT_REPORT.md): Full comprehensive forensic diagnostic report.
- [`checkpoint_comparison.csv`](file:///c:/Users/devin/MLUA/outputs/experiments/EXP-MLUA-001/diagnostics/e19_e22/checkpoint_comparison.csv): Metadata comparison of E19_BEST and E22_LATEST checkpoints.
- [`threshold_sweep.csv`](file:///c:/Users/devin/MLUA/outputs/experiments/EXP-MLUA-001/diagnostics/e19_e22/threshold_sweep.csv): Offline threshold sensitivity analysis from $\tau=0.05$ to $\tau=0.75$.
- [`per_case_comparison.csv`](file:///c:/Users/devin/MLUA/outputs/experiments/EXP-MLUA-001/diagnostics/e19_e22/per_case_comparison.csv): Per-case validation performance comparison across 50 cases.
- [`probability_statistics.csv`](file:///c:/Users/devin/MLUA/outputs/experiments/EXP-MLUA-001/diagnostics/e19_e22/probability_statistics.csv): Continuous logits and post-sigmoid probability distributions for foreground vs background.
- [`training_dynamics.csv`](file:///c:/Users/devin/MLUA/outputs/experiments/EXP-MLUA-001/diagnostics/e19_e22/training_dynamics.csv): Training metrics trajectory from E15 to E22.
- [`root_cause_analysis.json`](file:///c:/Users/devin/MLUA/outputs/experiments/EXP-MLUA-001/diagnostics/e19_e22/root_cause_analysis.json): Structured JSON summarizing root causes and decision support answers.
- `visuals/`: Diagnostic panels for 5 representative cases:
  - `case_1_high_degradation_23.png`: E19 Good $\to$ E22 Degraded.
  - `case_2_both_bad_2.png`: Both E19 & E22 Low Performance.
  - `case_3_e22_best_relative_2.png`: E22 Retained / Better Relative.
  - `case_4_high_prob_low_dice_17.png`: E22 High Peak Prob, Low Dice.
  - `case_5_strong_suppression_41.png`: Strongest Foreground Area Suppression.
