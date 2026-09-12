# MLUA Architecture Forensic Audit

## 1. Network Topology Overview

The official MLUA architecture (`model/FPN.py`) implements a **ResNet-34 Feature Pyramid Network (FPN) with Multi-Level Auxiliary Segmentation Heads and a Fused Segmentation Head**.

```
INPUT (1 × 384 × 384)
  │
  ▼
ResNet-34 Encoder (depth=5, pretrained='imagenet')
  ├── c1: Conv1 + BN + ReLU                  [B, 64, 192, 192]   (stride 2)
  ├── c2: MaxPool + Layer1                   [B, 64,  96,  96]   (stride 4)
  ├── c3: Layer2                             [B, 128, 48,  48]   (stride 8)
  ├── c4: Layer3                             [B, 256, 24,  24]   (stride 16)
  └── c5: Layer4                             [B, 512, 12,  12]   (stride 32)
  │
  ▼
FPN Top-Down Feature Pyramid Decoder (pyramid_channels = 256)
  ├── p5: Conv1x1(512 → 256) (c5)                                 [B, 256, 12, 12]
  ├── p4: Nearest2x(p5) + Conv1x1(256 → 256)(c4)                  [B, 256, 24, 24]
  ├── p3: Nearest2x(p4) + Conv1x1(128 → 256)(c3)                  [B, 256, 48, 48]
  └── p2: Nearest2x(p3) + Conv1x1(64  → 256)(c2)                  [B, 256, 96, 96]
  │
  ▼
Multi-Scale Segmentation Blocks (segmentation_channels = 128)
  ├── seg_block[0](p5): 3 × [Conv3x3-GN32-ReLU + Bilinear2x]      → f_p[0]: [B, 128, 96, 96]
  ├── seg_block[1](p4): 2 × [Conv3x3-GN32-ReLU + Bilinear2x]      → f_p[1]: [B, 128, 96, 96]
  ├── seg_block[2](p3): 1 × [Conv3x3-GN32-ReLU + Bilinear2x]      → f_p[2]: [B, 128, 96, 96]
  └── seg_block[3](p2): 1 × [Conv3x3-GN32-ReLU] (no upsample)     → f_p[3]: [B, 128, 96, 96]
  │
  ├──► MERGE BLOCK ('add'): sum(f_p[0], f_p[1], f_p[2], f_p[3])   → [B, 128, 96, 96]
  │      └──► Dropout2d(p = 0.2, inplace = True)                  → [B, 128, 96, 96]
  │             └──► SegmentationHead: Conv1x1(128 → 1) + Bilinear4x
  │                    └──► FUSED OUTPUT (masks)                  → [B, 1, 384, 384]
  │
  └──► AUXILIARY SEGMENTATION HEADS (4 heads)
         ├── AuxHead[0](f_p[0]): Conv1x1(128 → 1) + Bilinear4x    → [B, 1, 384, 384]
         ├── AuxHead[1](f_p[1]): Conv1x1(128 → 1) + Bilinear4x    → [B, 1, 384, 384]
         ├── AuxHead[2](f_p[2]): Conv1x1(128 → 1) + Bilinear4x    → [B, 1, 384, 384]
         └── AuxHead[3](f_p[3]): Conv1x1(128 → 1) + Bilinear4x    → [B, 1, 384, 384]
```

---

## 2. Component Specifications

### 2.1 Encoder
- **Base Architecture**: `resnet34` via `segmentation_models_pytorch.encoders.get_encoder`.
- **Input Channels**: 1 (Grayscale X-ray patch).
- **Depth**: 5 stages (`c0` to `c5`).
- **Initial Conv**: `Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)`.
- **Pretrained Weights**: `'imagenet'`.
- **Stage Output Channels**: `[64, 64, 128, 256, 512]`.

### 2.2 Decoder Blocks
- **Pyramid Channels**: 256.
- **Top-down fusion**:
  - `p5`: 1x1 Conv with 512 in_channels, 256 out_channels.
  - `p4`, `p3`, `p2`: 1x1 skip Conv projecting encoder channels to 256, followed by elementwise addition with `nearest` 2x upsampled higher-level pyramid feature.
- **Segmentation Blocks**:
  - Each block projects pyramid channels (256) to segmentation channels (128).
  - Uses `Conv3x3GNReLU`: `Conv2d(in_c, out_c, 3x3, stride=1, padding=1, bias=False)` + `GroupNorm(32, out_c)` + `ReLU(inplace=True)`.
  - Upsampling inside `Conv3x3GNReLU` uses `F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)`.
  - Number of upsamples per pyramid level:
    - Level 5 (`p5`): 3 upsamples (12x12 → 24x24 → 48x48 → 96x96).
    - Level 4 (`p4`): 2 upsamples (24x24 → 48x48 → 96x96).
    - Level 3 (`p3`): 1 upsample (48x48 → 96x96).
    - Level 2 (`p2`): 0 upsamples (already 96x96).

### 2.3 Head Architecture & Merge Policy
- **Merge Policy**: `"add"` (Elementwise addition of 4 feature maps of shape `[B, 128, 96, 96]`).
- **Regularization**: `nn.Dropout2d(p=0.2, inplace=True)`.
- **Fused Segmentation Head**: `Conv2d(128, 1, kernel_size=1, padding=0, bias=True)` + `UpsamplingBilinear2d(scale_factor=4)`.
- **Auxiliary Heads**: 4 identical heads, each `Conv2d(128, 1, kernel_size=1, padding=0, bias=True)` + `UpsamplingBilinear2d(scale_factor=4)` applied directly to individual `feature_pyramid` elements `f_p[0..3]`.

---

## 3. Parameter Count & Dimensions

| Submodule | Input Tensor | Output Tensor | Kernel / Operation | Approx. Parameters |
|---|---|---|---|---|
| **ResNet-34 Encoder** | `[B, 1, 384, 384]` | `c1..c5` | Conv7x7, ResBlocks | 21,284,672 |
| **p5 Lateral Conv** | `[B, 512, 12, 12]` | `[B, 256, 12, 12]` | Conv 1x1 | 131,328 |
| **p4 Skip Conv** | `[B, 256, 24, 24]` | `[B, 256, 24, 24]` | Conv 1x1 | 65,792 |
| **p3 Skip Conv** | `[B, 128, 48, 48]` | `[B, 256, 48, 48]` | Conv 1x1 | 33,024 |
| **p2 Skip Conv** | `[B, 64, 96, 96]` | `[B, 256, 96, 96]` | Conv 1x1 | 16,640 |
| **SegBlock 0 (p5)** | `[B, 256, 12, 12]` | `[B, 128, 96, 96]` | 3x (Conv3x3-GN32) | 590,592 |
| **SegBlock 1 (p4)** | `[B, 256, 24, 24]` | `[B, 128, 96, 96]` | 2x (Conv3x3-GN32) | 442,880 |
| **SegBlock 2 (p3)** | `[B, 256, 48, 48]` | `[B, 128, 96, 96]` | 1x (Conv3x3-GN32) | 295,168 |
| **SegBlock 3 (p2)** | `[B, 256, 96, 96]` | `[B, 128, 96, 96]` | 1x (Conv3x3-GN32) | 295,168 |
| **Fused Head** | `[B, 128, 96, 96]` | `[B, 1, 384, 384]` | Conv 1x1 + Bilinear4x | 129 |
| **4 Aux Heads** | 4x `[B, 128, 96, 96]` | 4x `[B, 1, 384, 384]` | 4x (Conv 1x1 + Bilinear4x) | 516 |
| **TOTAL** | | | | **~23,155,909 (~23.16M)** |

---

## 4. Alternative Implementations in Repository
- **`model/smpFPN.py` (`FPNnet`)**:
  - Implements a self-contained custom `ResNetEncoder` and `FPNDecoder`.
  - Used by `urpc_run.py`.
  - Contains subtle differences in normalization (BatchNorm2d instead of GroupNorm32) and head organization.
  - Used for URPC baseline experiments.
- **`clcc_run.py` & `uamt_run.py`**:
  - Use `smp.Unet` with single prediction head and optional auxiliary projection head (`aux_proj` in CLCC).
