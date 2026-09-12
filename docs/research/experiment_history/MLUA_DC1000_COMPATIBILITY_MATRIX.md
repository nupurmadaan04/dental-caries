# MLUA vs DC1000 Compatibility Matrix

This document provides a line-by-line compatibility audit comparing official MLUA codebase requirements against the fresh `DC1000_dataset.zip` contents.

---

## 1. Comprehensive Compatibility Matrix

| Requirement / Dimension | MLUA Official Expectation | DC1000 Provided Content | Compatible? | Forensic Analysis & Adapter Action |
|---|---|---|---|---|
| **Training Image Format** | Grayscale PNG / PIL readable (`PIL.Image.open`) | Grayscale PNG (`Mode L`, 8-bit) | **YES** | Directly compatible with `Image.open`. |
| **Training Mask Format** | Binary mask `{0, 255}` or `{0, 1}` uint8 | Binary mask (`{0, 255}` uint8) | **YES** | Directly scaled to `[0.0, 1.0]` by `ToTensor()`. |
| **Training Image Resolution** | $384 \times 384$ pixels | $384 \times 384$ in `DC1000_dataset/train/images` | **YES** | Exactly matches `transize = 384`. |
| **Training Mask Resolution** | $384 \times 384$ pixels | $384 \times 384$ in `DC1000_dataset/train/labels` | **YES** | Exactly matches `transize = 384`. |
| **Training Channel Count** | 1 channel (Grayscale) | 1 channel (`Mode L`) | **YES** | Model encoder expects `in_channels = 1`. |
| **Training File Linking** | Identical base names `ID.png` | `train/images/ID.png` $\leftrightarrow$ `train/labels/ID.png` | **YES** | 2,389 images link 1:1 with 2,389 labels (IDs $1 \dots 2528$). |
| **Training Directory Layout** | `data/train/images`, `data/train/labels` | `DC1000_dataset/train/images`, `DC1000_dataset/train/labels` | **ADAPTER NEEDED** | Directory path mapping or extraction to `data/train/` required. |
| **Sorting / Integer IDs** | `sorted(..., key=lambda x: int(x...))` | Integer filenames (`1.png`, `2.png`, etc.) | **YES** | File names are cleanly numeric integers. |
| **Validation Image Format** | Cropped dental arch PNG ($768 \times 1536$) | Repository contains `dataset/test/images_cut` | **YES** | 100 benchmark dental arch crops at $768 \times 1536$ are present in repo. |
| **Validation Mask Format** | Cropped dental arch binary PNG ($768 \times 1536$) | Repository contains `dataset/test/labels_cut` | **YES** | 100 benchmark binary masks at $768 \times 1536$ are present in repo. |
| **Validation Tiling Logic** | 21 patches ($384 \times 384$ with $192$ stride) | $768 \times 1536 \to 3 \times 7 = 21$ patches | **YES** | Recomposition algorithm `recompone_overlap` matches exactly. |
| **Semi-Supervised Sampler** | `TwoStreamBatchSampler` (4 L + 4 UL) | 2,389 available patches partitioned by index | **YES** | Supports 10% (265 L / 2124 UL), 20% (530 L / 1859 UL), 50% (1325 L / 1064 UL). |
| **Multi-class Caries Annotations** | Not used in baseline binary MLUA | Provided in `colors_clean` (102, 153, 255) | **EXPANSION READY** | Available for future multi-class / severity grading studies. |

---

## 2. Incompatibilities & Required Adapters

1. **Path Mapping Adapter**:
   - `mlua_run.py` looks for `data/train/images` and `data/train/labels` in the working directory.
   - The archive stores these under `DC1000_dataset/train/images` and `DC1000_dataset/train/labels`.
   - *Adapter*: Configure a unified dataset directory path or extract `DC1000_dataset/train` to `data/train`.
2. **Missing Dataset Helper Modules**:
   - `mlua_run.py` attempts `from dataset import TrainDataset, ValDataset` and `from dataloader import TwoStreamBatchSampler`.
   - *Adapter*: Supply `dataset.py` and `dataloader.py` (restored from `clcc_run.py`).
3. **Cross-Platform Path Sorting**:
   - `mlua_run.py` used Windows backslash splits `x.split('\\')`.
   - *Adapter*: Use `pathlib.Path(x).stem` for OS-agnostic sorting.
