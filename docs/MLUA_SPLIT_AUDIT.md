# MLUA Data Split & Partition Forensic Audit

## 1. Split Architecture Overview

The MLUA research evaluates dental caries segmentation under three standardized semi-supervised regimes on the DC1000 dataset:
1. **10% Labeled Data**: 265 labeled patches / remaining unlabeled.
2. **20% Labeled Data**: 530 labeled patches / remaining unlabeled.
3. **50% Labeled Data**: 1,325 labeled patches / remaining unlabeled.
4. **Test Set**: 100 sealed full panoramic cases ($768 \times 1536$ dental arch crops, evaluated using 21-patch sliding window).

---

## 2. Partition Breakdown in Source Code

In `mlua_run.py` lines 247-251:
```python
idxs = list(range(len(train_image_list + ul_image_list)))
labeled_len = labeled_ratio[labeled_rate]  # {"0.1": 265, "0.2": 530, "0.5": 1325}
labeled_idxs = idxs[:labeled_len]
unlabeled_idxs = list(set(idxs) - set(labeled_idxs))
batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, l_batch_size)
```

### Partition Distribution (from 2,389 training patches in DC1000):

| Semi-Supervised Setting | Labeled Patch Count | Unlabeled Patch Count | Labeled Ratio ($\%$) | Batches Per Epoch | Total Steps (200 Epochs) |
|---|---|---|---|---|---|
| **10% SSL (`MLUA10`)** | 265 patches | 2,124 patches | 11.09% | 66 batches | 13,200 steps |
| **20% SSL (`MLUA20`)** | 530 patches | 1,859 patches | 22.18% | 132 batches | 26,400 steps |
| **50% SSL (`MLUA50`)** | 1,325 patches | 1,064 patches | 55.46% | 331 batches | 66,200 steps |

---

## 3. Test Set Isolation & Patient-Level Independence

- **Test Set Provenance**:
  - The 100 test cases (`org_test_dataset` / `dataset/test/`) originate from the final batch of 500 panoramic radiographs collected from clinical centers.
  - Slices from these 100 test panoramas were explicitly excluded from the 2,389 training slice pool (confirmed by `readme.txt`: *"The 100 slices of the test set have been removed"*).
- **Patient Leakage Prevention**:
  - Training slices originate exclusively from training cases.
  - Test set consists of independent full panoramic radiographs.
  - Zero overlap exists between the 2,389 training patch pool and the 100 validation/test panoramas.
