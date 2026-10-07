# Canonical Checkpoint Storage & Integrity Note

The canonical checkpoint file `EXP-MLUA-003_E75_BEST.pth` is a 370.8 MB binary model state dict. Due to GitHub's 100 MB per-file push limit and standard Git quota constraints, the binary file is maintained in the local review package and verified locally.

### Canonical Checkpoint Metadata
- **Filename**: `EXP-MLUA-003_E75_BEST.pth`
- **File Size**: `370,840,935 bytes` (~353.66 MB)
- **SHA-256 Hash**: `cb52ed6ec6911fab7f0081588fb63a6f0bad0e37b61c37f0ba9bd293a6bcedfb`
- **Training Epoch**: 75 (selected from 80-epoch training schedule at global step 9,900)
- **Architecture**: Dual-stream Student/Teacher ResNet-34 + Feature Pyramid Network (FPN)
- **Input Patch Size**: 384 × 384 pixels, stride 192 (21 patches per panoramic radiograph)
- **Operating Threshold**: $\tau = 0.50$

### Verification Results
- **Validation Dice Score**: `71.867%` (Validation Loss: 0.7254, IoU: 57.349%, Precision: 78.132%, Recall: 67.343%)
- **Sealed Test Evaluation (100 Cases)**: Macro Dice `50.147%`, Micro Dice `52.924%`, Specificity `99.873%`
- **Local Location**:
  - `outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E75_BEST.pth`
  - `FINAL_REVIEW_PACKAGE/09_TECHNICAL_EVIDENCE/checkpoint/EXP-MLUA-003_E75_BEST.pth`

Detailed architecture definitions and training loops are documented in:
- [fpn.py](file:///c:/Users/devin/MLUA/FINAL_REVIEW_PACKAGE/09_TECHNICAL_EVIDENCE/model_architecture/fpn.py)
- [train_exp003.py](file:///c:/Users/devin/MLUA/FINAL_REVIEW_PACKAGE/09_TECHNICAL_EVIDENCE/training_engine/train_exp003.py)
- [CHECKPOINT_REGISTRY.md](file:///c:/Users/devin/MLUA/FINAL_REVIEW_PACKAGE/05_RESULTS/important_metrics/CHECKPOINT_REGISTRY.md)
