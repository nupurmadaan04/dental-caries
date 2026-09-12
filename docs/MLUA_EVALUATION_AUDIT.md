# MLUA Evaluation Pipeline & Metric Forensics Audit

## 1. Evaluation Architecture

Validation and evaluation in the official MLUA codebase are implemented across two distinct files:
1. `evaluate/utils.py`: Recomposition and NumPy/MedPy metric functions.
2. `mlua_run.py`: LightningModule hooks `validation_step` and `on_validation_epoch_end`.

---

## 2. Sliding Window Panoramic Recomposition

Panoramic radiographs are too large for direct patch-based FPN inference without boundary artifacts. The official codebase uses a 50% overlap sliding window mechanism.

### 2.1 Forward Tiling (`extract_ordered_overlap`)
- Full image padded if dimensions not divisible by stride: `paint_border_overlap`.
- Extracts 384x384 patches with vertical stride 192 and horizontal stride 192.
- For standard `768 × 1536` cropped dental arch images:
  - Vertical slices: 3 positions (0, 192, 384).
  - Horizontal slices: 7 positions (0, 192, 384, 576, 768, 960, 1152).
  - Total: 21 patches per full case.

### 2.2 Recomposition Algorithm (`recompone_overlap`)
```python
def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):
    # preds shape: (21, 1, 384, 384)
    full_prob = np.zeros((N_full_imgs, 1, img_h, img_w))
    full_sum  = np.zeros((N_full_imgs, 1, img_h, img_w))
    k = 0
    for i in range(N_full_imgs):
        for h in range((img_h - patch_h) // stride_h + 1):
            for w in range((img_w - patch_w) // stride_w + 1):
                full_prob[i, :, h*stride_h : h*stride_h+patch_h, w*stride_w : w*stride_w+patch_w] += preds[k]
                full_sum[i,  :, h*stride_h : h*stride_h+patch_h, w*stride_w : w*stride_w+patch_w] += 1
                k += 1
    final_avg = full_prob / full_sum
    final_avg = np.clip(final_avg, 0.0, 1.0)
    return final_avg
```
- Averages overlapping predictions across the interior pixels (each interior pixel is predicted up to 4 times).

---

## 3. Metric Calculations

### 3.1 Validation Loop Metric Calculation (`mlua_run.py`)
In `mlua_run.py` lines 140-150:
- Ground truth threshold: `gt = (gt > 0.5)`
- Prediction threshold: `pred_imgs = np.array(pred_imgs > 0.5).squeeze()`
- Evaluated case-by-case using `medpy.metric.binary`:
  - **Dice / F1-Score**: `metric.binary.dc(pred_imgs, gt)`
  - **Jaccard / IoU**: `metric.binary.jc(pred_imgs, gt)`
  - **Sensitivity / Recall**: `metric.binary.sensitivity(pred_imgs, gt)`
  - **Precision**: `metric.binary.precision(pred_imgs, gt)`
  - **Specificity**: `metric.binary.specificity(pred_imgs, gt)`

### 3.2 Evaluation Script Metric Helper (`evaluate/utils.py:metric_calculate`)
```python
def metric_calculate(target: np.ndarray, prediction: np.ndarray):
    target = np.uint8(target.flatten() > 0.5)
    prediction = np.uint8(prediction.flatten() > 0.5)
    TP = (prediction * target).sum()
    FN = ((1 - prediction) * target).sum()
    TN = ((1 - prediction) * (1 - target)).sum()
    FP = (prediction * (1 - target)).sum()

    acc  = (TP + TN) / (TP + TN + FP + FN + 1e-4)
    iou  = TP / (TP + FP + FN + 1e-4)
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-4)
    pre  = TP / (TP + FP + 1e-4)
    spe  = TN / (FP + TN + 1e-4)
    sen  = TP / (TP + FN + 1e-4)

    return acc, iou, dice, pre, spe, sen
```

---

## 4. Evaluation Quirks & Edge Case Behaviors

1. **Validation Skipping**:
   - `mlua_run.py` only computes validation metrics when `self.current_epoch % 10 == 0 or self.current_epoch > 150`.
   - On non-validation epochs, `val_mean_dice` is logged as `0.0`.
2. **Hard-Coded Mean Denominator**:
   - In `on_validation_epoch_end`:
     ```python
     mean_dice = sum(self.eval_dict["dice"]) / 100
     ```
   - The divisor is hard-coded to `100` regardless of the actual length of `val_loader`! If the validation set has 50 cases, the mean metric will be halved!
3. **Empty Ground Truth / Negative Cases Handling**:
   - `medpy.metric.binary.dc` on an empty ground truth image (0 true caries pixels) with 0 predicted pixels returns `1.0` or raises `ZeroDivisionError` depending on medpy version.
   - In `metric_calculate`, adding `1e-4` to the denominator means if `TP=FP=FN=0`, `dice = 0 / 1e-4 = 0.0`.
4. **Validation Batch Size Constraint**:
   - Validation `DataLoader` must have `batch_size = 1` because one batch item represents 21 tiles corresponding to exactly one panoramic X-ray.
