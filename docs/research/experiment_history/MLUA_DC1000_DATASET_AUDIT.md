# MLUA DC1000 Dataset Forensic Audit

## 1. Archive Identity & Metadata
- **Archive Filename**: `DC1000_dataset.zip` (located in repository root `c:\Users\devin\MLUA\DC1000_dataset.zip`)
- **Archive Size**: 1,099,499,277 bytes (~1.02 GB)
- **Total Archive Entries**: 7,175 entries (7,158 PNG images/masks, 3 TXT files, directory markers)
- **Archive Provenance**: Matches the official DC1000 Google Drive and Baidu Cloud release links published in the official MLUA repository `README.md`.

---

## 2. Complete Archive Directory Hierarchy & File Counts

```text
DC1000_dataset.zip
└── DC1000_dataset/
    ├── org_train_dataset/
    │   ├── readme.txt                    (Metadata explaining 497 annotated panoramas & clean subsets)
    │   ├── images/                       (497 raw panoramic dental radiographs, 2943 × 1435, Mode L)
    │   ├── labels_clean/                 (493 cleaned binary masks: 0 bg, 255 caries, 2943 × 1435, Mode L)
    │   ├── colors_clean/                 (493 cleaned 3-tier severity masks: 102 sup, 153 med, 255 deep)
    │   └── colors_bydoctors/             (497 raw doctor annotations: 51, 102, 153, 204, 255)
    │
    ├── org_test_dataset/
    │   ├── readme.txt                    (Metadata explaining 100 benchmark test panoramas)
    │   ├── images/                       (100 raw panoramic radiographs, 2943 × 1435, Mode L)
    │   ├── labels/                       (100 binary ground truth masks, 2943 × 1435, Mode L)
    │   ├── colors/                       (100 3-tier severity masks: 102, 153, 255)
    │   └── colors_origin_bydoctor/       (100 original doctor color annotations)
    │
    └── train/
        ├── readme.txt                    (Metadata explaining the training slice generation)
        ├── images/                       (2,389 training image patches, 384 × 384, Mode L)
        └── labels/                       (2,389 binary training mask patches, 384 × 384, Mode L)
```

---

## 3. Detailed Folder Properties & Statistics

| Directory Inside ZIP | File Count | Modality / Format | Resolution (W × H) | Value Range / Encoding | Purpose in MLUA |
|---|---|---|---|---|---|
| `DC1000_dataset/train/images/` | 2,389 | Grayscale PNG (Mode L) | 384 × 384 | `[0, 255]` uint8 | Ready-to-train 384x384 patch inputs for `TrainDataset` |
| `DC1000_dataset/train/labels/` | 2,389 | Binary Mask PNG (Mode L) | 384 × 384 | `{0, 255}` uint8 | Ground truth segmentation masks for training |
| `DC1000_dataset/org_train_dataset/images/` | 497 | Grayscale PNG (Mode L) | 2943 × 1435 | `[0, 255]` uint8 | Full panoramic radiographs for training pool |
| `DC1000_dataset/org_train_dataset/labels_clean/` | 493 | Binary Mask PNG (Mode L) | 2943 × 1435 | `{0, 255}` uint8 | Full panoramic clean binary caries masks |
| `DC1000_dataset/org_train_dataset/colors_clean/` | 493 | Multiclass PNG (Mode P/L) | 2943 × 1435 | `{0, 102, 153, 255}` | Multiclass caries severity ground truth |
| `DC1000_dataset/org_test_dataset/images/` | 100 | Grayscale PNG (Mode L) | 2943 × 1435 | `[0, 255]` uint8 | Full raw test panoramas (matches `dataset/test/images`) |
| `DC1000_dataset/org_test_dataset/labels/` | 100 | Binary Mask PNG (Mode L) | 2943 × 1435 | `{0, 255}` uint8 | Full raw test masks (matches `dataset/test/labels`) |

---

## 4. Dataset Identity Verification

- **Verification Status**: **100% VERIFIED OFFICIAL DC1000 DATASET**
- **Evidence 1**: The 100 filenames in `DC1000_dataset/org_test_dataset/images/` (e.g. `1008.png`, `1009.png`, `1016.png`, ..., `996.png`) are **100% identical** to the 100 test images stored in `c:\Users\devin\MLUA\dataset\test\images\`.
- **Evidence 2**: The `readme.txt` files inside the archive explicitly document the creation of the 100 test cases and the extraction of the 2,389 training slices.
- **Evidence 3**: The archive was downloaded directly from the official repository's Google Drive source.
