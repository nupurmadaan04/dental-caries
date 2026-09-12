# MLUA Data Pipeline & DC1000 Dataset Forensic Audit

## 1. Overview & Data Provenance

The MLUA project targets dental caries segmentation in panoramic radiographs from the **DC1000** dataset (1000 panoramic radiographs).

### 1.1 Dataset Availability Breakdown
- **In-Repository Data**: Exactly 100 test samples in `dataset/test/` (both full panoramic images/masks and cropped 768x1536 dental arch versions).
- **External Training Data**: Not tracked in Git. Downloadable from:
  - Google Drive: `https://drive.google.com/file/d/1Xn1oGHvhGF9GbkcLEtCOV5QvWWqt1y62/view?usp=drive_link`
  - Baidu Cloud: `https://pan.baidu.com/s/1jRXsSQIr8mm3EyGYELv9kg?pwd=rsoc`
- **Expected Directory Structure in Training Code**:
  - Labeled images: `data/train/images/`
  - Labeled masks: `data/train/labels/`
  - Unlabeled images: `data/train/unlabel_images/images/`
  - Validation images: `../caries_data/Max100Dice/images_cut/`
  - Validation masks: `../caries_data/Max100Dice/labels_cut/`

---

## 2. Training Dataset Pipeline (`TrainDataset`)

*(Extracted from `clcc_run.py` lines 42-85, matching the implementation imported in `mlua_run.py`)*

### 2.1 Initialization
- Accepts `image_list`, `label_list`, `ul_image_list`, and `transize = 384`.
- Labeled pairs are stored as `[image_path, label_path]`.
- Unlabeled samples are stored as `[ul_image_path, None]`.

### 2.2 Transforms & Augmentation Pipeline
```
Raw Image (disk) / Label (disk)
  │
  ├──► Read via PIL.Image.open()
  │      - Unlabeled mask generated as np.zeros((384, 384))
  │
  ├──► Synchronized Spatial Transforms (both_transform):
  │      - Seed set to random.randint(0, 10000)
  │      - torch.random.manual_seed(seed)
  │      - RandomHorizontalFlip(p=0.5)
  │      - RandomRotation(degrees=45)
  │      (Applied synchronously to both image and label using same random seed)
  │
  ├──► Photometric Transforms (img_transform) [Image Only]:
  │      - ColorJitter(brightness=0.5, contrast=0.5)
  │
  ├──► Resizing:
  │      - T.Resize((384, 384)) on image and label
  │
  ├──► Tensor Conversion & Normalization:
  │      - T.ToTensor() (scales [0, 255] PIL image to [0.0, 1.0] float tensor)
  │      - torch.tensor(np.array(image), dtype=torch.float32)  → [1, 384, 384]
  │      - torch.tensor(np.array(label), dtype=torch.float32)  → [1, 384, 384]
```

---

## 3. Semi-Supervised TwoStreamBatchSampler

*(Extracted from `clcc_run.py` lines 138-156)*

```python
class TwoStreamBatchSampler(Sampler):
    def __init__(self, l_indices, ul_indices, batch_size, l_batch_size):
        self.l_indices = l_indices
        self.ul_indices = ul_indices
        self.l_batch_size = l_batch_size
        self.ul_batch_size = batch_size - l_batch_size
```

### 3.1 Batch Composition & Sampling Logic
- **Total Batch Size**: 8 (`batch_size = 8`).
- **Labeled Batch Size**: 4 (`l_batch_size = 4`).
- **Unlabeled Batch Size**: 4 (`ul_batch_size = 4`).
- **Sampling Behavior**:
  - `label_iter = iterate_once(self.l_indices)`: Permutes labeled indices once per epoch.
  - `unlabel_iter = iterate_eternally(self.ul_indices)`: Infinite shuffling stream of unlabeled indices.
  - Number of batches per epoch (`__len__`): `len(labeled_indices) // 4`.
  - Every minibatch tensor `[8, 1, 384, 384]` contains:
    - Indices `0..3`: Labeled patches (with valid ground truth).
    - Indices `4..7`: Unlabeled patches (with zero dummy masks).

### 3.2 Labeled Ratios & Partition Counts

| Partition Ratio | Labeled Count in Code (`labeled_ratio`) | Unlabeled Pool | Epoch Length (Batches) |
|---|---|---|---|
| **10% (`0.1`)** | 265 samples | 2,385 samples | 66 batches (265 // 4) |
| **20% (`0.2`)** | 530 samples | 2,120 samples | 132 batches (530 // 4) |
| **50% (`0.5`)** | 1,325 samples | 1,325 samples | 331 batches (1325 // 4) |

*Note on dataset counts: The partition dictionary indicates a total of 2,650 training patches generated from raw radiographs.*

---

## 4. Validation & Evaluation Pipeline (`ValDataset`)

*(Extracted from `clcc_run.py` lines 87-118 and `evaluate/utils.py`)*

### 4.1 Input Specification
- Inputs are full-size or cropped dental arch panoramic images (e.g. `768 × 1536`).
- Ground truth masks are corresponding `768 × 1536` binary arrays.

### 4.2 Overlapping Patch Extraction (`get_data_test_overlap`)
- **Full Image Size**: Height = 768, Width = 1536.
- **Patch Size**: 384 × 384.
- **Stride**: Height stride = 192, Width stride = 192 (50% overlap).
- **Number of Patches per Image**:
  - Vertical steps: `(768 - 384) // 192 + 1 = 3`.
  - Horizontal steps: `(1536 - 384) // 192 + 1 = 7`.
  - Total patches per panoramic image: `3 × 7 = 21 patches`.
- **Validation Batch Tensor**: `[21, 384, 384]`.
- **Dataloader Configuration**: `batch_size = 1` (loads 1 panoramic case = 21 patches).

### 4.3 Validation Step Execution
1. Dataloader yields `imgs: [1, 21, 384, 384]`, `gt: [1, 768, 1536]`.
2. Permute: `imgs = imgs.permute(1, 0, 2, 3)` → `[21, 1, 384, 384]`.
3. Model forward: `outputs = self(imgs)` (computes student fused logits `[21, 1, 384, 384]`).
4. Sigmoid activation: `pred = torch.sigmoid(outputs)`.
5. Reconstruction (`recompone_overlap`):
   - Reconstructs `[1, 1, 768, 1536]` probability map by summing overlapping patch predictions and dividing by the overlap count matrix.
6. Thresholding: `pred_imgs = (pred_imgs > 0.5)`.
7. Case-level metrics computed using `medpy.metric.binary`.
