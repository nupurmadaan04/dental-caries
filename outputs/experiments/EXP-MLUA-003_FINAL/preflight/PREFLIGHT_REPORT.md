# EXP-MLUA-003 Pre-Flight Validation Report

- **Validation Status**: `PASSED`
- **Baseline Experiment**: `EXP-MLUA-002` (Frozen at Epoch 9)
- **Controlled Experiment**: `EXP-MLUA-003`
- **Intended Difference**: Teacher BatchNorm buffer synchronization (`running_mean`, `running_var`, `num_batches_tracked`)
- **Unintended Differences**: `ZERO`

---

## 1. Configuration Parameter Audit

| Parameter Category | EXP-MLUA-002 Value | EXP-MLUA-003 Value | Status |
| :--- | :---: | :---: | :---: |
| `seed` | `42` | `42` | **Identical** |
| `dataset` | `DC1000` | `DC1000` | **Identical** |
| `labeled_count` | `530` | `530` | **Identical** |
| `unlabeled_count` | `1859` | `1859` | **Identical** |
| `batch_size` | `8 (4 labeled + 4 unlabeled)` | `8 (4 labeled + 4 unlabeled)` | **Identical** |
| `architecture` | `ResNet-34 + FPN (4 aux heads)` | `ResNet-34 + FPN (4 aux heads)` | **Identical** |
| `optimizer` | `AdamW (lr=0.001, wd=0.01)` | `AdamW (lr=0.001, wd=0.01)` | **Identical** |
| `scheduler` | `LambdaLR (poly=0.9)` | `LambdaLR (poly=0.9)` | **Identical** |
| `ssl.ema_theta` | `0.99` | `0.99` | **Identical** |
| `ssl.mc_iterations` | `8` | `8` | **Identical** |
| `ssl.noise_sigma` | `0.01` | `0.01` | **Identical** |
| `ssl.noise_clamp` | `0.1` | `0.1` | **Identical** |
| `ssl.consistency_weight_max` | `0.1` | `0.1` | **Identical** |
| `ssl.consistency_rampup_epochs` | `200` | `200` | **Identical** |
| `evaluation.decision_threshold` | `0.50` | `0.50` | **Identical** |

---

## 2. EXP-MLUA-002 Baseline Integrity & Sealed Test Set

- `EXP-MLUA-002_LATEST.pth` SHA256: `97cbec8d57361a248fd2b7468774bcf56f1b33f96b267a04270664d4c5d9ee2b`
- `EXP-MLUA-002_BEST.pth` SHA256: `f631733b13ecc7b79e042011f6b650b864387726a282447925cafe1a14d92e42`
- Sealed Test Set Cases: `100 images, 100 labels` (Untouched).

---

## 3. Pre-Training Buffer Synchronization Verification

- **Float Buffers (`running_mean`, `running_var`)**: Successfully synchronized via in-place EMA ($lpha = 0.99$).
- **Integer Buffers (`num_batches_tracked`)**: Successfully synchronized via in-place direct copy.
- **Teacher Gradients**: Verified 0 gradients (`requires_grad=False`).
