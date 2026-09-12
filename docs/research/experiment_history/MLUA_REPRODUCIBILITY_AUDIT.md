# MLUA Official Reproducibility Audit

## 1. Reproducibility Classification

**Classification**: **PARTIAL / DIFFICULT FROM REPOSITORY ALONE**

### Rationale
The repository contains the complete mathematical logic, architecture, loss formulations, and 100 sample evaluation pairs. However, out-of-the-box execution fails immediately if launched without addressing missing imports, dataset downloads, and OS-specific hard-coded paths.

---

## 2. Blocking Reproducibility Barriers

### 2.1 Missing Source Files (Import Errors)
- **Problem**: `mlua_run.py` attempts to import:
  ```python
  from dataset import TrainDataset, ValDataset
  from dataloader import TwoStreamBatchSampler
  ```
  Neither `dataset.py` nor `dataloader.py` exists in the repository.
- **Remedy**: The implementations are preserved inside `clcc_run.py` (lines 42-156) and can be extracted cleanly.

### 2.2 Absent Training Dataset
- **Problem**: Git repository only contains 100 test samples (`dataset/test/`). The training images (`data/train/images`, `data/train/labels`, `data/train/unlabel_images/images`) are not included.
- **Remedy**: Download full DC1000 dataset via Google Drive or Baidu Cloud links provided in `README.md`.

### 2.3 Hard-Coded Windows File Paths & Relative Path Assumptions
- **Problem**: `mlua_run.py` contains Windows-specific backslashes and relative paths:
  - `file_path = pwd + "\\data"`
  - `image_path = os.path.join(file_path, "train\\images")`
  - `panorama_gt_path = os.path.join(f_pwd, "caries_data\Max100Dice\labels_cut")` (note escape character typo `\M`)
  - Sorting keys assume Windows backslashes: `int(x.split('\\')[-1][:-4])`
  - TensorBoard logger: `pl_loggers.TensorBoardLogger('.\\Cariouslog\\MULA')`
- **Remedy**: Refactor to use `pathlib.Path` or `os.path` for cross-platform compatibility.

### 2.4 Hard-Coded Single GPU Index
- **Problem**:
  ```python
  gpu_list = [0]
  gpu_list_str = ','.join(map(str, gpu_list))
  os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
  trainer = Trainer(..., gpus=[0, ])
  ```
  Fails on multi-GPU nodes or CPU-only test environments.

### 2.5 PyTorch Lightning Version Compatibility
- **Problem**: The codebase was authored with PyTorch Lightning v1.x (`gpus=[0, ]`, `precision=16`). PyTorch Lightning v2.x requires `accelerator="gpu"`, `devices=[0]`, and updated validation hooks (`on_validation_epoch_end`).

---

## 3. Reproducibility Checklist & Resolution Matrix

| Factor | Repository Status | Required Action for Full Standalone Reimplementation |
|---|---|---|
| **Model Code (`model/FPN.py`)** | Complete & Functional | Preserve exactly; add clean type annotations. |
| **Loss & Uncertainty Logic** | Complete & Functional | Preserve exact mathematical equations and weighting. |
| **Dataset Loader Code** | Missing `dataset.py` | Restore from `clcc_run.py` into modular package. |
| **Batch Sampler** | Missing `dataloader.py` | Restore `TwoStreamBatchSampler` from `clcc_run.py`. |
| **Training Data** | Not in Git | Ingest from DC1000 dataset archive. |
| **Test Data** | Present (100 pairs) | Available in `dataset/test/` for testing sliding-window evaluation. |
| **Pretrained Weights** | Dynamic download | Automatically pulled via `smp` or local cache. |
| **Environment Specs** | Missing requirements file | Create pinned `requirements.txt` and `pyproject.toml`. |
| **OS Compatibility** | Windows hard-coded | Implement `pathlib` for Linux/macOS/Windows support. |
