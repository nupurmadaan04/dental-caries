# MLUA Bug & Implementation Quirk Forensic Audit

This document cataloging all bugs, quirks, and anomalies discovered during the forensic audit of the official MLUA repository.

---

## 1. Summary Table of Discovered Quirks & Bugs

| ID | Finding | Location | Severity | Impact | Affects Reported Metrics? |
|---|---|---|---|---|---|
| **BUG-01** | Missing `dataset.py` & `dataloader.py` | `mlua_run.py:30-31` | **CRITICAL** | Immediate `ModuleNotFoundError` on script run | Yes (prevents execution) |
| **BUG-02** | Hard-coded Windows path separators & split logic | `mlua_run.py:226-240` | **HIGH** | Crash on Linux/macOS environments | Yes (prevents execution on non-Windows) |
| **BUG-03** | Typo in raw path string escape (`\Max100Dice`) | `mlua_run.py:240` | **MEDIUM** | Invalid string parse if run without raw string | Potential path error |
| **BUG-04** | Hard-coded mean metric divisor (`/ 100`) | `mlua_run.py:154-158` | **HIGH** | Incorrect validation metric if test set $\neq 100$ | Yes (distorts reported numbers) |
| **BUG-05** | Entropy calculation along single channel (`dim=1`) | `mlua_run.py:100` | **MEDIUM** | Uses $-2p\ln p$ instead of $-p\ln p - (1-p)\ln(1-p)$ | Minor (threshold was tuned to this specific curve) |
| **BUG-06** | Deep supervision loss normalized by 4 instead of 5 | `mlua_run.py:119` | **LOW** | Constant scale factor $1.25\times$ on supervised loss | No (loss magnitude adjusted during training) |
| **BUG-07** | Factor of 2 in consistency loss denominator | `mlua_run.py:104` | **LOW** | Constant scale factor $0.5\times$ on consistency loss | No (absorbed into $\lambda$) |
| **BUG-08** | Fixed divisor in UAMT script local variable bug | `uamt_run.py:188` | **HIGH** | In `uamt_run.py`, `para1 = ...` fails in-place update | Affects UAMT baseline comparison |
| **BUG-09** | Validation metric logging set to 0.0 on skipped epochs | `mlua_run.py:166-170` | **LOW** | TensorBoard curves show dips to 0.0 on non-val epochs | Visualization artifact only |
| **BUG-10** | Unused command-line arguments (`--sigma`) | `mlua_run.py:40` | **INFORMATIONAL** | `--sigma` parsed but never passed to model | None |
| **BUG-11** | Unused imports in `mlua_run.py` & `util/utils.py` | Multiple files | **INFORMATIONAL** | `KFold`, `albumentations`, `thop` imported but unused | None |

---

## 2. Detailed Technical Breakdown

### BUG-01: Missing `dataset.py` & `dataloader.py`
- **Location**: `mlua_run.py` lines 30-31:
  ```python
  from dataset import TrainDataset, ValDataset
  from dataloader import TwoStreamBatchSampler
  ```
- **Analysis**: The author modularized dataset utilities during local development into `dataset.py` and `dataloader.py`, but omitted them from Git. The identical code was left inline in `clcc_run.py`.

### BUG-02: Hard-Coded Backslashes in Sorting Key
- **Location**: `mlua_run.py` lines 232-233:
  ```python
  train_image_list = sorted(train_image_list, key=lambda x: int(x.split('\\')[-1][:-4]), reverse=False)
  ```
- **Analysis**: On POSIX systems (Linux/macOS), `x.split('\\')` returns the full path as a single string. `int(x[:-4])` then attempts to convert the entire path string into an integer and raises a `ValueError`.

### BUG-04: Hard-Coded Divisor in `on_validation_epoch_end`
- **Location**: `mlua_run.py` lines 154-158:
  ```python
  mean_iou = sum(self.eval_dict["iou"]) / 100
  mean_dice = sum(self.eval_dict["dice"]) / 100
  mean_spe = sum(self.eval_dict["spe"]) / 100
  mean_pre = sum(self.eval_dict["pre"]) / 100
  mean_sen = sum(self.eval_dict["sen"]) / 100
  ```
- **Analysis**: The author hard-coded `100` because the validation set had exactly 100 cases (`Max100Dice`). If evaluated on a different number of cases (e.g. 50 or 200), the calculated mean metric is mathematically wrong. Must be `sum(...) / len(self.eval_dict["dice"])`.

### BUG-05: Single-Sided Channel Entropy Formulation
- **Location**: `mlua_run.py` line 100:
  ```python
  uncertainty = -2.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
  ```
- **Analysis**: Since `preds` has shape `[4, 1, 384, 384]`, `dim=1` is a singleton channel dimension. The standard binary entropy is $H(p) = -p\ln p - (1-p)\ln(1-p)$. The code computes $-2p\ln p$. Because $-2p\ln p$ is monotonically decreasing above $p = 1/e \approx 0.368$, higher confidence caries predictions have low values below the ramp-up threshold $\beta(t) \in [0.52, 0.69]$, which filters uncertain boundary pixels.

### BUG-08: Local Variable Reassignment in `uamt_run.py`
- **Location**: `uamt_run.py` line 188:
  ```python
  for para1, para2 in zip(self.model_stu.parameters(), self.model_tea.parameters()):
      para1 = alpha * para1 + (1 - alpha) * para2
  ```
- **Analysis**: In Python, assigning `para1 = ...` merely rebinds the local loop variable without modifying the tensor in-place. Zzz512 correctly fixed this in `mlua_run.py` (commit `5b0941b`) using `para1.data = ...`, but left the broken baseline in `uamt_run.py`.
