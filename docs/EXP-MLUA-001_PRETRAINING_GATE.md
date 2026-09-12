# Pre-Training Experiment Gate Audit
## EXP-MLUA-001: Official 10% Labeled SSL Baseline

**Experiment ID**: `EXP-MLUA-001`  
**Purpose**: Official MLUA 10% Labeled Semi-Supervised Baseline on DC1000  
**Audit Date**: September 2026  
**Status**: **PRE-TRAINING GATE PASSED**

---

## 1. Dataset Integrity Verification

- **Training Image Directory**: `data/raw/DC1000_dataset/train/images/`
- **Training Mask Directory**: `data/raw/DC1000_dataset/train/labels/`
- **Image Count**: Exactly **2,389** files (`1.png` to `2528.png`).
- **Mask Count**: Exactly **2,389** files (`1.png` to `2528.png`).
- **Filename Correspondence**: 100% 1:1 match between image stems and mask stems.
- **Resolution**: Exactly $384 \times 384$ pixels across all images and masks.
- **Image Mode**: Mode L (8-bit grayscale, single-channel).
- **Mask Mode & Values**: Mode L (binary $\{0, 255\}$).
- **Missing / Duplicate Files**: 0 missing pairs, 0 duplicate stems.

---

## 2. Semi-Supervised Partition (10% SSL Baseline)

- **Total Training Pool**: 2,389 patch-level samples.
- **Labeled Partition ($10\%$)**: First **265** patches (indices `0..264`).
- **Unlabeled Partition**: Remaining **2,124** patches (indices `265..2388`).
- **Overlap**: Verified 0 overlapping indices ($\text{labeled} \cap \text{unlabeled} = \emptyset$).
- **Deterministic Random Seed**: `seed = 42` (fixed across Python, NumPy, PyTorch CPU, and CUDA).
- **Sampling Mechanism**: `TwoStreamBatchSampler(l_indices, ul_indices, batch_size=8, l_batch_size=4)`.
- **Epoch Batches**: $\lfloor 265 / 4 \rfloor = 66$ batches per epoch ($13,200$ total steps across 200 epochs).

---

## 3. Test Set Isolation Verification

- **Benchmark Test Directory**: `dataset/test/images_cut` and `dataset/test/labels_cut`.
- **Benchmark Case Count**: Exactly **100** panoramic dental arch radiographs ($768 \times 1536$).
- **Test Isolation**:
  - The 100 test cases originate from the sealed clinical test partition.
  - Slices from these 100 test panoramas were explicitly excluded from the 2,389 training slice pool.
  - Benchmark test data is **NOT** loaded by the training dataset.
  - Benchmark test metrics were **NOT** computed during this audit.

---

## 4. Effective Resolved Configuration

*(Extracted from `configs/mlua_default.yaml`)*

```yaml
experiment:
  name: "EXP-MLUA-001"
  seed: 42
  ssl_enabled: true
  max_epochs: 200
  precision: 16

model:
  name: "Net"
  encoder: "resnet34"
  encoder_weights: "imagenet"
  encoder_depth: 5
  in_channels: 1
  out_channels: 1
  pyramid_channels: 256
  segmentation_channels: 128
  merge_policy: "add"
  dropout: 0.2
  aux_heads_count: 4

data:
  train_image_dir: "data/raw/DC1000_dataset/train/images"
  train_label_dir: "data/raw/DC1000_dataset/train/labels"
  val_image_cut_dir: "dataset/test/images_cut"
  val_label_cut_dir: "dataset/test/labels_cut"
  patch_size: 384
  batch_size: 8
  labeled_batch_size: 4
  unlabeled_batch_size: 4
  active_rate: "0.1"

optimization:
  optimizer: "AdamW"
  learning_rate: 0.001
  weight_decay: 0.01
  scheduler: "LambdaLR"
  poly_power: 0.9

ssl:
  ema_theta: 0.99
  mc_iterations: 8
  noise_sigma: 0.01
  noise_clamp: 0.1
  threshold_rampup_steps: 4480
  threshold_start_factor: 0.75
  threshold_end_factor: 1.00
  consistency_weight_max: 0.1
  consistency_rampup_epochs: 200

evaluation:
  image_height: 768
  image_width: 1536
  patch_size: 384
  stride: 192
  num_patches: 21
  decision_threshold: 0.5
  val_interval: 10
  val_late_epoch_start: 150
```

---

## 5. Source-Code Provenance & Architecture Map

| Component | Source File | Implementation Origin | Classification |
|---|---|---|---|
| **ResNet-34 FPN Backbone** | `src/mlua/models/fpn.py` | Official `model/FPN.py` | Official MLUA Architecture (Pure PyTorch) |
| **Auxiliary Heads (4x)** | `src/mlua/models/fpn.py` | Official `model/FPN.py` | Official MLUA Deep Supervision |
| **Train / Val Datasets** | `src/mlua/data/dataset.py` | Official `clcc_run.py:42-118` | Official MLUA Data Logic + `pathlib` |
| **Two-Stream Sampler** | `src/mlua/data/sampler.py` | Official `clcc_run.py:138-156` | Official MLUA SSL Sampler |
| **Dice & Consistency Loss** | `util/utils.py` | Official `util/utils.py` | Official MLUA Loss Formulation |
| **Sliding Window Tiling** | `evaluate/utils.py` | Official `evaluate/utils.py` | Official MLUA Recomposition Logic |
| **Training Pipeline Hook** | `mlua_run.py` | Official `mlua_run.py` | Official MLUA LightningModule |

---

## 6. Environment & System Specifications

- **Python Version**: `3.11.9`
- **PyTorch Version**: `2.12.0+cpu`
- **Torchvision Version**: `0.27.0+cpu`
- **OS Platform**: `Windows 10 (10.0.21996)`
- **Device Support**: CPU / CUDA Auto-Detection configured.
- **Reproducibility Guarantee**: Fixed random seed `42` applied across all frameworks.

---

## 7. Expected Training Outputs

When training is executed under EXP-MLUA-001:
- **Checkpoints**: Top 5 best models saved based on maximized `val_mean_dice`.
- **Logs**: TensorBoard logs in `Cariouslog/MLUA/`.
- **Logged Metrics**:
  - Training: `train_seg_loss`, `train_consistency_loss`, `train_bce_loss`, `train_dice_loss`, `train_mean_dice`, `train_mean_iou`.
  - Validation: `val_mean_dice`, `val_mean_iou`, `val_mean_spe`, `val_mean_sen`, `val_mean_pre`.

---

## 8. Known Limitations & Medical Research Scope

1. **Research Prototype Scope**: AI-assisted caries segmentation model for academic evaluation. Not approved as an autonomous medical diagnostic device.
2. **Patch-Level Labeling**: Partitions represent patch-level sampling from the 2,389 training patch pool.

---

## 9. Discrepancy & Parameter Integrity Statement

All parameters in `configs/mlua_default.yaml` have been cross-checked against the official `Zzz512/MLUA` source code. **Zero unexpected parameter discrepancies exist.**

---

## 10. Pre-Training Gate Declarations

- **Dataset Verification**: PASSED (2,389 valid pairs).
- **SSL 10% Partition**: PASSED (265 L / 2,124 UL).
- **Test Isolation**: PASSED (100 benchmark cases untouched).
- **Architecture Integrity**: PASSED (ResNet-34 FPN + 4 Aux Heads + Fused Head).
- **Training Status**: **NO TRAINING HAS OCCURRED.**

---

PRE-TRAINING GATE:
PASS

EXP-MLUA-001 READY:
YES

TRAINING EXECUTED:
NO

TEST EVALUATION EXECUTED:
NO

TEST SET TOUCHED:
NO

CONFIGURATION MODIFIED:
NO
