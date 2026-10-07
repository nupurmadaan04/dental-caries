# Model Checkpoints Directory & Registry

All model checkpoints are canonically managed, version-controlled by SHA-256 hash, and stored in their respective experiment directories under `outputs/experiments/`:

## 1. Active Canonical Production Checkpoint
- **Experiment**: `EXP-MLUA-003` (Extended Run to Epoch 78, 10,296 steps)
- **Active Model**: `EXP-MLUA-003_E75_BEST.pth` (Peak Validation Epoch 75)
- **Canonical Path**: `outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E75_BEST.pth`
- **File Size**: 370,840,935 bytes
- **SHA-256 Hash**: `cb52ed6ec6911fab7f0081588fb63a6f0bad0e37b61c37f0ba9bd293a6bcedfb`
- **Operating Threshold**: $\tau = 0.50$
- **Val Dice / Loss**: 71.867% / 0.7254
- **Final Sealed-Test Macro Dice**: 50.147% (100 panoramic radiographs)

## 2. Preserved Historical Baselines
- **Historical Milestone Reference (Epoch 64)**:
  - **Path**: `outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E64_BEST.pth`
  - **SHA-256 Hash**: `167918b8375815668b5e784902e3e82a865b86053a610a312c775d52d473c526`
  - **Val Dice / Loss**: 69.386% / 0.7471
- **Historical Baseline (Epoch 56)**:
  - **Path**: `outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E56_FINAL.pth`
  - **Val Dice / Loss**: 65.623% / 0.7639
- **Supervised Baseline (`EXP-MLUA-001`)**:
  - **Path**: `outputs/experiments/EXP-MLUA-001_HISTORICAL/checkpoints/EXP-MLUA-001_BEST.pth`
  - **SHA-256 Hash**: `e718e29a0cd6e09738b8bcdf79436dbf8a9cb31356de2601c3ac2cb68cd86b77`
  - **Val Dice / Loss**: 54.21% / 0.8920

---
*Note: This registry ensures a single source of truth across all training, inference, and evaluation pipelines.*
