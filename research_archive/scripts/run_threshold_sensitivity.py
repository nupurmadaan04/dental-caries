import os
import sys
import yaml
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base_dir = Path("c:/Users/devin/MLUA")
sys.path.insert(0, str(base_dir))

from src.mlua.models.fpn import Net

# 1. Verification of Checkpoint
best_ckpt_path = base_dir / "outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E56_FINAL.pth"
assert best_ckpt_path.exists(), "BEST checkpoint does not exist!"

print("[1] Loading BEST Checkpoint...", flush=True)
ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
epoch = ckpt.get("epoch")
step = ckpt.get("global_step")
metrics_logged = ckpt.get("metrics")
print(f"    Loaded Epoch: {epoch}, Step: {step}", flush=True)
print(f"    Logged Checkpoint Metrics: {metrics_logged}", flush=True)

assert epoch == 56, f"Expected Epoch 56, got {epoch}"
assert step == 7392, f"Expected Step 7392, got {step}"

# Check finiteness
for k, v in ckpt["model_stu_state_dict"].items():
    if v.dtype.is_floating_point:
        assert torch.isfinite(v).all(), f"NaN/Inf in student parameter {k}"
print("    All model parameters are 100% finite and verified.", flush=True)

# 2. Reconstruct Model & Validation Dataset exactly as in train_exp003.py
config_file = base_dir / "configs/experiments/exp_mlua_003_final_config.yaml"
with open(config_file, "r") as f:
    cfg = yaml.safe_load(f)

model_stu = Net(in_c=1, out_c=1, encoder_weights=None)
model_stu.load_state_dict(ckpt["model_stu_state_dict"])
model_stu.eval()

train_img_dir = base_dir / cfg["data"]["train_image_dir"]
train_lbl_dir = base_dir / cfg["data"]["train_label_dir"]
train_img_files = sorted(list(train_img_dir.glob("*.png")), key=lambda x: int(x.stem))
train_lbl_files = sorted(list(train_lbl_dir.glob("*.png")), key=lambda x: int(x.stem))

labeled_count = cfg["data"]["labeled_count"] # 530
all_indices = list(range(len(train_img_files)))
labeled_indices = all_indices[:labeled_count]
val_indices = labeled_indices[:50]

class CachedRawDataset:
    def __init__(self, image_list, label_list, transize=384):
        self.transize = transize
        self.cached_images = []
        self.cached_labels = []
        for img_path, lbl_path in zip(image_list, label_list):
            img = Image.open(str(img_path)).convert("L")
            if lbl_path is not None and lbl_path.exists():
                lbl = Image.open(str(lbl_path)).convert("L")
            else:
                lbl = Image.fromarray(np.zeros((transize, transize), dtype=np.uint8))
            self.cached_images.append(img)
            self.cached_labels.append(lbl)

class DeterministicSubsetDataset(Dataset):
    def __init__(self, parent_dataset, indices):
        self.parent = parent_dataset
        self.indices = indices
        self.resize_transform = T.Resize((parent_dataset.transize, parent_dataset.transize))
        self.normalize_transform = T.ToTensor()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image = self.parent.cached_images[real_idx].copy()
        label = self.parent.cached_labels[real_idx].copy()

        image = self.resize_transform(image)
        label = self.resize_transform(label)

        image_tensor = self.normalize_transform(image)
        label_tensor = self.normalize_transform(label)
        return image_tensor, label_tensor

print(f"[2] Pre-caching {len(train_img_files)} dataset items into RAM...", flush=True)
raw_dataset = CachedRawDataset(train_img_files, train_lbl_files, transize=cfg["data"]["patch_size"])
val_dataset = DeterministicSubsetDataset(raw_dataset, val_indices)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

# 3. Single-Pass Inference to collect all Sigmoid Probability Maps & GTs
print("[3] Running single-pass inference on validation set (50 patches)...", flush=True)
all_pred_sigs = []
all_gts = []

with torch.no_grad():
    for v_imgs, v_gts in val_loader:
        if v_imgs.ndim == 3: v_imgs = v_imgs.unsqueeze(1)
        if v_gts.ndim == 3: v_gts = v_gts.unsqueeze(1)
        v_pred_fused, _ = model_stu(v_imgs)
        v_pred_sig = torch.sigmoid(v_pred_fused)
        all_pred_sigs.append(v_pred_sig)
        all_gts.append(v_gts)

pred_probs = torch.cat(all_pred_sigs, dim=0) # [50, 1, 384, 384]
gt_masks = torch.cat(all_gts, dim=0)         # [50, 1, 384, 384]
print(f"    Collected validation predictions: {pred_probs.shape}, GT: {gt_masks.shape}", flush=True)

# 4. Metric function matching calculate_metrics_batch from train_exp003.py exactly
def eval_at_threshold(pred_probs: torch.Tensor, gts: torch.Tensor, threshold: float):
    # Batch-by-batch averaging as done in train_exp003.py (batch_size=4, 13 batches: 12 of 4, 1 of 2)
    batch_size = 4
    n_samples = pred_probs.size(0)
    
    dices, ious, precs, recs, specs, f1s, prevs = [], [], [], [], [], [], []
    zero_pred_count = 0
    
    for i in range(0, n_samples, batch_size):
        p_b = pred_probs[i:i+batch_size]
        g_b = gts[i:i+batch_size]
        
        preds = (p_b > threshold).float()
        g_bin = (g_b > 0.5).float()
        
        tp = (preds * g_bin).sum().item()
        fp = (preds * (1.0 - g_bin)).sum().item()
        fn = ((1.0 - preds) * g_bin).sum().item()
        tn = ((1.0 - preds) * (1.0 - g_bin)).sum().item()
        
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        f1 = dice
        pred_prev = preds.mean().item()
        
        dices.append(dice)
        ious.append(iou)
        precs.append(prec)
        recs.append(rec)
        specs.append(spec)
        f1s.append(f1)
        prevs.append(pred_prev)
        
        for b_i in range(preds.size(0)):
            if preds[b_i].sum().item() == 0:
                zero_pred_count += 1
                
    return {
        "threshold": threshold,
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
        "specificity": float(np.mean(specs)),
        "pred_prevalence": float(np.mean(prevs)),
        "zero_pred_ratio": float(zero_pred_count / n_samples)
    }

# 5. Baseline Verification at tau = 0.50
baseline = eval_at_threshold(pred_probs, gt_masks, threshold=0.50)
print("\n--- BASELINE REPRODUCIBILITY CHECK (tau = 0.50) ---", flush=True)
print(f"Computed Dice:        {baseline['dice']*100:.3f}% (Expected: ~65.623%)")
print(f"Computed IoU:         {baseline['iou']*100:.3f}% (Expected: ~49.854%)")
print(f"Computed Precision:   {baseline['precision']*100:.3f}% (Expected: ~69.009%)")
print(f"Computed Recall:      {baseline['recall']*100:.3f}% (Expected: ~63.649%)")
print(f"Computed Specificity: {baseline['specificity']*100:.3f}% (Expected: ~99.753%)")
print(f"Zero-Pred Ratio:      {baseline['zero_pred_ratio']*100:.1f}% (Expected: 0.0%)")

diff_dice = abs(baseline['dice'] - metrics_logged['val_dice'])
assert diff_dice < 1e-4, f"Metric mismatch! Computed {baseline['dice']} vs Logged {metrics_logged['val_dice']}"
print(">>> BASELINE REPRODUCIBILITY VERIFIED 100.0% EXACT MATCH <<<\n", flush=True)

# 6. Sweep 19 Thresholds: 0.05 to 0.95 in 0.05 increments
thresholds = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]
results = []

for t in thresholds:
    res = eval_at_threshold(pred_probs, gt_masks, threshold=t)
    results.append(res)
    print(f"Threshold tau={t:.2f} -> Dice: {res['dice']*100:.3f}%, IoU: {res['iou']*100:.3f}%, Prec: {res['precision']*100:.3f}%, Rec: {res['recall']*100:.3f}%, Spec: {res['specificity']*100:.3f}%, ZeroPred: {res['zero_pred_ratio']*100:.1f}%", flush=True)

df_sweep = pd.DataFrame(results)

# 7. Save Artifacts in outputs/diagnostics/EXP-MLUA-003_VALIDATION_THRESHOLD_ANALYSIS/
diag_dir = base_dir / "outputs/diagnostics/EXP-MLUA-003_VALIDATION_THRESHOLD_ANALYSIS"
diag_dir.mkdir(parents=True, exist_ok=True)
csv_out = diag_dir / "EXP-MLUA-003_THRESHOLD_SWEEP.csv"
df_sweep.to_csv(csv_out, index=False)
print(f"\n[+] Saved sweep CSV to {csv_out.as_posix()}", flush=True)

# 8. Generate Curves Plot
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

ax1.plot(df_sweep["threshold"], df_sweep["dice"] * 100, label="Validation Dice (%)", color="#1f77b4", linewidth=2.5, marker="o")
ax1.plot(df_sweep["threshold"], df_sweep["precision"] * 100, label="Precision (%)", color="#2ca02c", linewidth=2.0, linestyle="--", marker="s")
ax1.plot(df_sweep["threshold"], df_sweep["recall"] * 100, label="Recall (%)", color="#d62728", linewidth=2.0, linestyle="-.", marker="^")
ax1.plot(df_sweep["threshold"], df_sweep["iou"] * 100, label="IoU / Jaccard (%)", color="#9467bd", linewidth=1.5, linestyle=":", marker="d")

# Highlight tau=0.50
base_dice = baseline["dice"] * 100
ax1.axvline(x=0.50, color="gray", linestyle="--", alpha=0.7, label=f"Canonical tau=0.50 (Dice={base_dice:.2f}%)")

# Highlight peak Dice
best_row = df_sweep.loc[df_sweep["dice"].idxmax()]
ax1.scatter([best_row["threshold"]], [best_row["dice"] * 100], color="#ff7f0e", s=150, zorder=5, label=f"Peak Dice @ tau={best_row['threshold']:.2f} ({best_row['dice']*100:.2f}%)")

ax1.set_xlabel("Decision Threshold (tau)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
ax1.set_title("EXP-MLUA-003 Validation Threshold Sensitivity Curve (Epoch 56 BEST Checkpoint)", fontsize=13, fontweight="bold", pad=12)
ax1.set_xticks(thresholds)
ax1.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=45)
ax1.set_ylim(0, 105)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="lower left", fontsize=10, framealpha=0.95)
fig.tight_layout()

plot_out = diag_dir / "EXP-MLUA-003_THRESHOLD_CURVE.png"
fig.savefig(plot_out)
plt.close(fig)
print(f"[+] Saved visualization plot to {plot_out.as_posix()}", flush=True)

# 9. Print Key Comparison Findings
best_dice_res = df_sweep.loc[df_sweep["dice"].idxmax()]
best_prec_res = df_sweep.loc[df_sweep["precision"].idxmax()]
best_rec_res = df_sweep.loc[df_sweep["recall"].idxmax()]

print("\n=======================================================", flush=True)
print(f"OPTIMAL THRESHOLD FINDINGS (EPOCH 56 BEST):", flush=True)
print(f"  Best Dice Threshold:      tau = {best_dice_res['threshold']:.2f} -> Dice = {best_dice_res['dice']*100:.3f}%, Prec = {best_dice_res['precision']*100:.3f}%, Rec = {best_dice_res['recall']*100:.3f}%, IoU = {best_dice_res['iou']*100:.3f}%")
print(f"  Best Precision Threshold: tau = {best_prec_res['threshold']:.2f} -> Prec = {best_prec_res['precision']*100:.3f}%, Dice = {best_prec_res['dice']*100:.3f}%, Rec = {best_prec_res['recall']*100:.3f}%")
print(f"  Best Recall Threshold:    tau = {best_rec_res['threshold']:.2f} -> Rec = {best_rec_res['recall']*100:.3f}%, Dice = {best_rec_res['dice']*100:.3f}%, Prec = {best_rec_res['precision']*100:.3f}%")
print(f"  Delta Dice vs tau=0.50:   {best_dice_res['dice']*100 - baseline['dice']*100:+.3f} percentage points")
print(f"  Delta Prec vs tau=0.50:   {best_dice_res['precision']*100 - baseline['precision']*100:+.3f} percentage points")
print(f"  Delta Rec vs tau=0.50:    {best_dice_res['recall']*100 - baseline['recall']*100:+.3f} percentage points")
print("=======================================================\n", flush=True)
