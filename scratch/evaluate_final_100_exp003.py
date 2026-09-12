import os
import sys
import json
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

base_dir = Path("c:/Users/devin/MLUA")
sys.path.insert(0, str(base_dir))

from src.mlua.models.fpn import Net
from src.mlua.evaluation.final_100_evaluation import (
    extract_panoramic_patches,
    reconstruct_panoramic,
    calculate_case_metrics,
)

# 1. Verification of Target Checkpoint
best_ckpt_path = base_dir / "outputs/experiments/EXP-MLUA-003/checkpoints/EXP-MLUA-003_BEST.pth"
assert best_ckpt_path.exists(), f"Checkpoint missing: {best_ckpt_path}"

print(f"[1] Loading Best Checkpoint from {best_ckpt_path.as_posix()}...", flush=True)
ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)

epoch = ckpt.get("epoch")
step = ckpt.get("global_step")
val_metrics = ckpt.get("metrics")
print(f"    Epoch: {epoch}, Global Step: {step}", flush=True)
print(f"    Preserved Validation Metrics: {val_metrics}", flush=True)

assert epoch == 56, f"Expected Epoch 56, found {epoch}"
assert step == 7392, f"Expected Step 7392, found {step}"

# Verify parameter finiteness
for k, v in ckpt["model_stu_state_dict"].items():
    if v.dtype.is_floating_point:
        assert torch.isfinite(v).all(), f"NaN/Inf in student parameter: {k}"
print("    Model parameters: 100% FINITE and VERIFIED.", flush=True)

# 2. Build Model
model = Net(in_c=1, out_c=1, encoder_name="resnet34", encoder_weights=None)
model.load_state_dict(ckpt["model_stu_state_dict"])
model.eval()
print("[2] Net (ResNet-34 + FPN MLUA) instantiated and loaded in eval mode.", flush=True)

# 3. Locate Sealed 100-Case Test Set
test_img_dir = base_dir / "dataset/test/images_cut"
test_lbl_dir = base_dir / "dataset/test/labels_cut"

test_img_files = sorted(list(test_img_dir.glob("*.png")), key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem)
test_lbl_files = sorted(list(test_lbl_dir.glob("*.png")), key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem)

assert len(test_img_files) == 100, f"Expected 100 test images, found {len(test_img_files)}"
assert len(test_lbl_files) == 100, f"Expected 100 test labels, found {len(test_lbl_files)}"
print(f"[3] Located {len(test_img_files)} sealed panoramic test cases in {test_img_dir.as_posix()}", flush=True)

# 4. Perform Single-Pass Independent Evaluation at tau = 0.50
threshold = 0.50
patch_size = 384
stride = 192
H, W = 768, 1536

case_results = []
zero_pred_cases = 0

print(f"[4] Running 21-patch sliding window inference (tau={threshold:.2f})...", flush=True)

for idx, (img_path, lbl_path) in enumerate(zip(test_img_files, test_lbl_files)):
    case_id = img_path.stem
    
    # Load raw image & label (strictly read-only)
    img = Image.open(str(img_path))
    lbl = Image.open(str(lbl_path)).convert("L")
    
    img_np = np.asarray(img) # [768, 1536] or [768, 1536, 3]
    lbl_np = np.asarray(lbl) # [768, 1536]
    
    # Extract 21 overlapping 384x384 patches
    patches, coords = extract_panoramic_patches(img_np, patch_size=patch_size, stride=stride)
    
    # Tensorize
    patch_tensors = []
    for p in patches:
        p_norm = p.astype(np.float32) / 255.0 if p.max() > 1.0 else p.astype(np.float32)
        if p_norm.ndim == 2:
            p_norm = np.expand_dims(p_norm, 0)
        elif p_norm.ndim == 3 and p_norm.shape[2] == 3:
            p_norm = np.mean(p_norm, axis=2, keepdims=True).transpose(2, 0, 1)
        patch_tensors.append(torch.from_numpy(p_norm))
    
    batch_tensor = torch.stack(patch_tensors, dim=0) # [21, 1, 384, 384]
    
    with torch.inference_mode():
        pred_fused, _ = model(batch_tensor)
        pred_probs = torch.sigmoid(pred_fused).squeeze(1).cpu().numpy() # [21, 384, 384]
        
    recon_prob = reconstruct_panoramic(list(pred_probs), coords, full_shape=(H, W), patch_size=patch_size)
    
    metrics = calculate_case_metrics(recon_prob, lbl_np, threshold=threshold, case_id=case_id)
    case_results.append(metrics)
    
    if metrics["pred_foreground_pixels"] == 0:
        zero_pred_cases += 1
        
    if (idx + 1) % 20 == 0 or (idx + 1) == 100:
        print(f"    Evaluated {idx+1:03d}/100 cases | Current Case {case_id}: Dice={metrics['dice']*100:.2f}%, Prec={metrics['precision']*100:.2f}%, Rec={metrics['recall']*100:.2f}%", flush=True)

# 5. Compute Macro & Micro Aggregations
df_cases = pd.DataFrame(case_results)

# Macro Aggregation
macro_dice = float(df_cases["dice"].mean())
macro_iou = float(df_cases["iou"].mean())
macro_prec = float(df_cases["precision"].mean())
macro_rec = float(df_cases["recall"].mean())
macro_spec = float(df_cases["specificity"].mean())
macro_f1 = float(df_cases["f1"].mean())
zero_pred_ratio = float(zero_pred_cases / len(df_cases))

# Micro Aggregation
total_tp = int(df_cases["tp"].sum())
total_fp = int(df_cases["fp"].sum())
total_fn = int(df_cases["fn"].sum())
total_tn = int(df_cases["tn"].sum())
eps = 1e-6

micro_dice = float((2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn + eps))
micro_iou = float(total_tp / (total_tp + total_fp + total_fn + eps))
micro_prec = float(total_tp / (total_tp + total_fp + eps))
micro_rec = float(total_tp / (total_tp + total_fn + eps))
micro_spec = float(total_tn / (total_tn + total_fp + eps))
micro_f1 = micro_dice

# 6. Save Artifacts in outputs/diagnostics/EXP-MLUA-003_FINAL_TEST/
out_dir = base_dir / "outputs/diagnostics/EXP-MLUA-003_FINAL_TEST"
out_dir.mkdir(parents=True, exist_ok=True)

per_case_csv = out_dir / "EXP-MLUA-003_FINAL_TEST_PER_CASE.csv"
df_cases.to_csv(per_case_csv, index=False)

summary_rows = [
    {
        "evaluation_scope": "EXP-MLUA-003 Final Sealed Test Set (100 Cases)",
        "aggregation_type": "MACRO (Case Mean)",
        "dice": round(macro_dice, 5),
        "iou": round(macro_iou, 5),
        "precision": round(macro_prec, 5),
        "recall": round(macro_rec, 5),
        "specificity": round(macro_spec, 5),
        "f1": round(macro_f1, 5),
        "zero_pred_case_ratio": round(zero_pred_ratio, 4),
        "total_cases": 100,
        "threshold": threshold,
    },
    {
        "evaluation_scope": "EXP-MLUA-003 Final Sealed Test Set (100 Cases)",
        "aggregation_type": "MICRO (Global Pixel Sum)",
        "dice": round(micro_dice, 5),
        "iou": round(micro_iou, 5),
        "precision": round(micro_prec, 5),
        "recall": round(micro_rec, 5),
        "specificity": round(micro_spec, 5),
        "f1": round(micro_f1, 5),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_tn": total_tn,
        "total_cases": 100,
        "threshold": threshold,
    },
]
df_summary = pd.DataFrame(summary_rows)
summary_csv = out_dir / "EXP-MLUA-003_FINAL_TEST_SUMMARY.csv"
df_summary.to_csv(summary_csv, index=False)

print("\n=======================================================", flush=True)
print("FINAL SEALED TEST SET EVALUATION RESULTS (100 CASES):", flush=True)
print(f"  MACRO Test Dice:        {macro_dice*100:.3f}%")
print(f"  MACRO Test IoU:         {macro_iou*100:.3f}%")
print(f"  MACRO Test Precision:   {macro_prec*100:.3f}%")
print(f"  MACRO Test Recall:      {macro_rec*100:.3f}%")
print(f"  MACRO Test Specificity: {macro_spec*100:.3f}%")
print(f"  Zero-Pred Case Ratio:   {zero_pred_ratio*100:.1f}%")
print(f"  MICRO Test Dice:        {micro_dice*100:.3f}%")
print(f"  MICRO Test IoU:         {micro_iou*100:.3f}%")
print(f"  MICRO Test Precision:   {micro_prec*100:.3f}%")
print(f"  MICRO Test Recall:      {micro_rec*100:.3f}%")
print(f"  MICRO Test Specificity: {micro_spec*100:.3f}%")
print(f"  Total Pixels:           TP={total_tp}, FP={total_fp}, FN={total_fn}, TN={total_tn}")
print("=======================================================\n", flush=True)
