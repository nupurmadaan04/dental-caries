# EXP-MLUA-001 Validation Cleanliness Fix Report

---

## 1. Executive Summary

This report documents the validation-pipeline cleanliness correction applied to **EXP-MLUA-001** before resuming training from Epoch 12.

**Correction Applied:** Replaced stochastic validation loading (which previously inherited `RandomRotation(45)`, `RandomHorizontalFlip(p=0.5)`, and `ColorJitter` from `train_dataset`) with a dedicated `DeterministicSubsetDataset` that evaluates **strictly deterministic, unaugmented raw patches** with standard $(384 \times 384)$ resizing and `ToTensor` normalization.

**Sanity Check Verification:** A read-only evaluation on the exact same Epoch 11 checkpoint confirmed that validation metrics are computed cleanly with zero stochastic variance, and that the model's underlying prediction distribution ($p_{\max} \approx 0.4035 < 0.50$) is preserved.

---

## 2. Validation Cleanliness Modifications

| Component | Old Implementation | Corrected Implementation | Scientific Status |
|---|---|---|---|
| **Validation Dataset Class** | `torch.utils.data.Subset(train_dataset, val_indices)` | `DeterministicSubsetDataset(train_dataset, val_indices)` | Fixed |
| **Spatial Transforms** | `RandomRotation(45)` + `RandomHorizontalFlip(p=0.5)` | **None** (Deterministic Raw Orientation) | Fixed |
| **Photometric Transforms** | `ColorJitter(brightness=0.5, contrast=0.5)` | **None** (Raw Grayscale Values) | Fixed |
| **Normalization & Sizing** | `Resize(384, 384)` + `ToTensor()` | `Resize(384, 384)` + `ToTensor()` | Unchanged |
| **Validation Samples** | 50 labeled patches (`labeled_indices[:50]`) | 50 labeled patches (`labeled_indices[:50]`) | Unchanged |
| **Decision Threshold** | `0.50` | `0.50` | Unchanged |
| **Training Pipeline** | Dynamic spatial + photometric augmentations | Dynamic spatial + photometric augmentations | **100% UNCHANGED** |

---

## 3. Read-Only Comparison on Epoch 11 Checkpoint

Evaluated across all 50 validation patches using `EXP-MLUA-001_LATEST.pth` (Epoch 11):

| Metric | Old Stochastic Validation | New Deterministic Validation | Difference / Note |
|---|---:|---:|---|
| **Val Loss** | `1.04299` | `1.04252` | Stable loss baseline |
| **Val Dice ($\tau = 0.50$)** | `0.00000` | `0.00000` | Identical (Under-detection regime at E11) |
| **Val IoU ($\tau = 0.50$)** | `0.00000` | `0.00000` | Identical |
| **Val Precision ($\tau = 0.50$)** | `0.00000` | `0.00000` | Identical |
| **Val Recall ($\tau = 0.50$)** | `0.00000` | `0.00000` | Identical |
| **Val Specificity ($\tau = 0.50$)** | `1.00000` | `1.00000` | Identical (100% Background Identification) |
| **Val F1 ($\tau = 0.50$)** | `0.00000` | `0.00000` | Identical |
| **Max Predicted Probability** | `0.45656` | `0.40352` | Clean unaugmented peak probability |
| **Mean Predicted Probability** | `0.01515` | `0.01686` | Clean background probability |

---

## 4. Scientific Integrity & Safety Confirmation

The following parameters are **100% FROZEN AND PRESERVED**:
- **Architecture:** ResNet-34 FPN with 4 auxiliary heads + 1 fused head (unchanged).
- **Dataset Split:** DC1000 10% labeled partition (265 Labeled / 2,124 Unlabeled) (unchanged).
- **Optimizer & LR:** AdamW (lr=0.001, weight_decay=0.01), Polynomial LR decay (unchanged).
- **EMA Teacher:** $\alpha = 0.99$ (unchanged).
- **Monte Carlo Uncertainty:** $T=8$, Gaussian $\sigma=0.01$, clamp $[-0.1, 0.1]$, dynamic threshold schedule (unchanged).
- **Loss Formulation:** Deep supervision (0.5 BCE + 0.5 Dice across all heads) + Sigmoid MSE Consistency (unchanged).
- **Training Augmentation:** Preserved in full for `train_loader`.
- **Checkpoints:** `EXP-MLUA-001_LATEST.pth` and `EXP-MLUA-001_BEST.pth` are unmodified.
- **Sealed Test Set (`dataset/test/`):** **100% UNTOUCHED & SEALED**.

---

## 5. Ready Status

- **Validation Cleanliness:** **`RESOLVED (100% DETERMINISTIC)`**
- **Epoch 11 State:** Intact in `EXP-MLUA-001_LATEST.pth`
- **Training Status:** **`PAUSED & READY TO RESUME FROM EPOCH 12`**
