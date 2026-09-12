# MLUA Official Repository Forensic Snapshot Audit

## 1. Repository Identity & Provenance
- **Official Repository URL**: `https://github.com/Zzz512/MLUA.git`
- **Paper**: *"Multi-level Uncertainty Aware Semi-supervised Learning for Dental Panoramic Caries Segmentation"*, Neurocomputing, Vol. 540, 2023, 126208.
- **Repository Owner**: Zzz512 (Authors: Xianyun Wang, Sizhe Gao, Kaisheng Jiang, Huicong Zhang, Linhong Wang, Feng Chen, Jun Yu, Fan Yang)
- **Active Branch**: `main`
- **Head Commit SHA**: `17b73ac9a7cbf2e293ca0a835b3ee28e67a4d531`
- **Total Commit Count**: 18 commits
- **Commit History Overview**:
  - `17b73ac` (Zzz512) Update README.md
  - `f72afdd` (Zzz512) Update README.md
  - `cd7e632` (Zzz512) Update README.md
  - `448353c` (Zzz512) Update README.md
  - `d442544` (Zzz512) Update README.md
  - `61705ba` (Zzz512) Update README.md
  - `1fc85ba` (Zzz512) Update README.md
  - `85a70ca` (Zzz512) Update README.md
  - `3e34445` (Zzz512) Update README.md
  - `5b0941b` (Zzz512) Update mlua_run.py (Fixed student/teacher EMA and model invocation)
  - `0fbfc42` (Zzz512) Update README.md
  - `ba323a7` (kkcr) util tool upload (Added evaluate/common.py and evaluate/utils.py)
  - `27c07f5` (kkcr) Merge branch 'main' of https://github.com/Zzz512/MLUA
  - `ba2b223` (kkcr) main code update (Uploaded clcc_run.py, mlua_run.py, uamt_run.py, urpc_run.py, model/FPN.py, model/smpFPN.py, util/utils.py)
  - `1d73773` (Zzz512) Update README.md
  - `19314d4` (kkcr) test dataset update (Uploaded dataset/test with 100 panoramic images/labels)
  - `cd56d89` (Zzz512) Update README.md
  - `47596d0` (Zzz512) Initial commit

---

## 2. Complete Repository Tree Structure

```text
c:\Users\devin\MLUA/
├── .git/
├── README.md                                 (1,438 bytes, dataset download links & paper citation)
├── mlua_run.py                               (12,150 bytes, official MLUA training script)
├── clcc_run.py                               (20,375 bytes, CLCC benchmark SSL script)
├── uamt_run.py                               (11,993 bytes, UAMT benchmark SSL script)
├── urpc_run.py                               (12,123 bytes, URPC benchmark SSL script)
├── dataset/
│   └── test/
│       ├── colors/                           (100 files: colored RGB overlay visualizations)
│       ├── colors_cut/                       (100 files: cropped tooth region colored overlays)
│       ├── images/                           (100 files: full panoramic X-ray images, PNG)
│       ├── images_cut/                       (100 files: cropped dental arch panoramic images, 768x1536 PNG)
│       ├── labels/                           (100 files: full panoramic ground truth binary masks)
│       └── labels_cut/                       (100 files: cropped dental arch binary masks, 768x1536 PNG)
├── evaluate/
│   ├── common.py                             (3,495 bytes: evaluation helper functions)
│   ├── utils.py                              (9,813 bytes: sliding window overlap extraction & metrics)
│   └── __pycache__/
│       ├── common.cpython-38.pyc
│       ├── metrics_us.cpython-38.pyc
│       ├── panorama_eval.cpython-38.pyc
│       ├── test.cpython-38.pyc
│       ├── utils.cpython-38.pyc
│       └── utils.cpython-38_????_Yukki_20220930001924.pyc
├── model/
│   ├── FPN.py                                (6,115 bytes: official MLUA ResNet-34 FPN architecture)
│   ├── smpFPN.py                             (23,306 bytes: custom FPN implementation for URPC)
│   └── __pycache__/
│       ├── FPN.cpython-38.pyc
│       ├── FPNDecoder.cpython-38.pyc
│       ├── smpFPN.cpython-38.pyc
│       └── smpUNet.cpython-38.pyc
└── util/
    ├── utils.py                              (15,308 bytes: loss functions, ramp-up schedules, metric helpers)
    └── __pycache__/
        ├── utils.cpython-38.pyc
        └── utils.cpython-38_????_Yukki_20221002135037.pyc
```

---

## 3. Artifact Categorization

### 1. Official Source Code
- `mlua_run.py`: Primary MLUA LightningModule and training orchestration.
- `model/FPN.py`: Primary MLUA ResNet-34 Feature Pyramid Network with 4 auxiliary heads and 1 fused head.
- `util/utils.py`: `DiceLoss`, `sigmoid_mse_loss`, `sigmoid_rampup`, `get_current_consistency_weight`, `mean_metric`.
- `evaluate/utils.py`: `recompone_overlap`, `metric_calculate`, `get_data_test_overlap`, `paint_border_overlap`, `extract_ordered_overlap`.
- `evaluate/common.py`: `readImg`, `readLabel`, `AverageMeter`, `weight_initV1`, `save_args`.

### 2. Test / Sample Data (Included in Repo)
- `dataset/test/`: Contains exactly 100 test samples in 6 representations:
  - `images`: Full raw panoramic dental radiographs.
  - `labels`: Full raw caries segmentation masks.
  - `images_cut`: Cropped dental arch images (standardized to 768x1536 resolution).
  - `labels_cut`: Cropped dental arch masks (standardized to 768x1536 resolution).
  - `colors`: Full panoramic image with colored mask overlay.
  - `colors_cut`: Cropped dental arch image with colored mask overlay.

### 3. Training Data (Absent from Repository)
- The training data is **NOT stored in git**.
- Expected local training path referenced in scripts:
  - `data/train/images`
  - `data/train/labels`
  - `data/train/unlabel_images/images`
- Download links provided in `README.md`:
  - Google Drive: `https://drive.google.com/file/d/1Xn1oGHvhGF9GbkcLEtCOV5QvWWqt1y62/view?usp=drive_link`
  - Baidu Cloud: `https://pan.baidu.com/s/1jRXsSQIr8mm3EyGYELv9kg?pwd=rsoc`

### 4. Generated Artifacts & PyCache
- Precompiled `.pyc` files from Python 3.8 are present in `evaluate/__pycache__`, `model/__pycache__`, `util/__pycache__`.
- Includes author development artifacts such as `utils.cpython-38_????_Yukki_20220930001924.pyc`.

### 5. Missing Code Files Referenced by Import Statements
- **`dataset.py`**:
  - `mlua_run.py`, `uamt_run.py`, `urpc_run.py` all execute `from dataset import TrainDataset, ValDataset`.
  - `dataset.py` is missing from the repository root.
  - *Note*: `TrainDataset` and `ValDataset` were embedded directly in `clcc_run.py` (lines 42-118).
- **`dataloader.py`**:
  - `mlua_run.py`, `uamt_run.py`, `urpc_run.py` execute `from dataloader import TwoStreamBatchSampler`.
  - `dataloader.py` is missing from the repository root.
  - *Note*: `TwoStreamBatchSampler` was embedded directly in `clcc_run.py` (lines 138-156).

### 6. External Dependencies
- `torch`, `torchvision`
- `pytorch-lightning` (LightningModule, Trainer, ModelCheckpoint, LearningRateMonitor)
- `segmentation-models-pytorch` (encoder extraction `get_encoder("resnet34")`)
- `albumentations`
- `medpy` (metric computation `binary.dc`, `binary.jc`, `binary.sensitivity`, etc.)
- `scikit-learn` (`KFold`)
- `opencv-python` (`cv2`)
- `pillow` (`PIL.Image`)
- `thop` (optional, FLOPs/params profiling in `util/utils.py`)
