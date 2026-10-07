# Dataset Storage and Download Note

The original DC1000 dataset archive is not stored in ordinary Git because of its size (1,099,499,277 bytes / ~1.10 GB), exceeding standard repository storage constraints.

The local final-review package contains the verified original archive:
- **Local Path**: `FINAL_REVIEW_PACKAGE/02_DATASET/DC1000_dataset.zip`
- **Source Archive**: `data/raw/DC1000_dataset.zip`
- **File Size**: `1,099,499,277 bytes` (1.02 GiB)
- **Content**: 1,000 raw panoramic radiographs (`.png`), 1,000 ground-truth caries annotation masks (`.png`), and official partition splits.

Detailed dataset analysis, structure, class distribution, and research considerations are fully documented in:
- [DATASET_README.md](file:///c:/Users/devin/MLUA/FINAL_REVIEW_PACKAGE/02_DATASET/DATASET_README.md)
- [DATASET_ANALYSIS.md](file:///c:/Users/devin/MLUA/FINAL_REVIEW_PACKAGE/02_DATASET/DATASET_ANALYSIS.md)
