# EXP-MLUA-003 Official Training Status Dashboard
**Controlled Remediation Experiment: Teacher BatchNorm Buffer Synchronization**

- **Experiment Status**: `STOPPED_CLEANLY`
- **Last Status Update**: `2026-09-12 15:06:02`
- **Completed Epochs**: `60 / 200` (30.0%)
- **Best Validation Dice**: `65.623%` (Epoch 56)
- **Current Validation Dice**: `61.527%`
- **Zero-Prediction Patch Ratio**: `0.0%`
- **Critical E10 Batch 68 Region**: `PASSED 100% FINITE`
- **Cumulative Training Time**: `133068.8s` (~`36.96 hours`)
- **Checkpoints Location**: `C:/Users/devin/MLUA/outputs/experiments/EXP-MLUA-003/checkpoints`

---

## Controlled Intervention Integrity
- **Baseline**: `EXP-MLUA-002` (Frozen at Epoch 9)
- **Single Change**: Teacher EMA synchronizes `model_tea.buffers()` (`running_mean`, `running_var`, `num_batches_tracked`) alongside `model_tea.parameters()`.
- **All Other Variables**: 100% identical (DC1000 dataset, seed 42, 530 labeled / 1859 unlabeled, ResNet-34 + FPN, pure FP32, sealed test set untouched).
