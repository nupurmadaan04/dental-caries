import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

base_dir = Path("c:/Users/devin/MLUA")
sys.path.insert(0, str(base_dir))

from src.mlua.models.fpn import Net
from src.mlua.evaluation.final_100_evaluation import (
    extract_panoramic_patches,
    reconstruct_panoramic,
)

print("[1] Loading Checkpoint...", flush=True)
ckpt_path = base_dir / "outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E56_FINAL.pth"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

epoch = ckpt.get("epoch")
step = ckpt.get("global_step")
val_metrics = ckpt.get("metrics")
print(f"    Epoch: {epoch}, Step: {step}", flush=True)
print(f"    Preserved Validation Metrics: {val_metrics}", flush=True)

nan_params = 0
inf_params = 0
for k, v in ckpt["model_stu_state_dict"].items():
    if v.dtype.is_floating_point:
        nan_params += int(torch.isnan(v).sum().item())
        inf_params += int(torch.isinf(v).sum().item())

print(f"    Parameter Health: NaN count={nan_params}, Inf count={inf_params}", flush=True)
assert nan_params == 0 and inf_params == 0, "Non-finite model parameters detected!"

# 2. Build Model
print("[2] Instantiating ResNet-34 + FPN MLUA...", flush=True)
model = Net(in_c=1, out_c=1, encoder_name="resnet34", encoder_weights=None)
load_res = model.load_state_dict(ckpt["model_stu_state_dict"], strict=True)
print(f"    Load state dict result: {load_res}", flush=True)
model.eval()

# 3. Load Representative NON-TEST Image
print("[3] Loading representative training image (101.png)...", flush=True)
rep_img_path = base_dir / "data/raw/DC1000_dataset/org_train_dataset/images/101.png"
assert rep_img_path.exists(), f"Missing representative image: {rep_img_path}"

raw_img = Image.open(rep_img_path).convert("L")
orig_w, orig_h = raw_img.size
print(f"    Raw Image Dimensions: {orig_w}x{orig_h}", flush=True)

# Resize to standard panoramic format (1536x768)
panoramic_img = raw_img.resize((1536, 768), Image.Resampling.BILINEAR)
img_np = np.asarray(panoramic_img)
H, W = img_np.shape
print(f"    Standard Panoramic Input Array: {H}x{W}, dtype={img_np.dtype}, min={img_np.min()}, max={img_np.max()}", flush=True)

# 4. Extract 21 overlapping 384x384 patches (stride 192)
print("[4] Extracting 21 overlapping 384x384 patches (stride 192)...", flush=True)
patches, coords = extract_panoramic_patches(img_np, patch_size=384, stride=192)
print(f"    Extracted patch count: {len(patches)}", flush=True)

patch_tensors = []
for p in patches:
    p_norm = p.astype(np.float32) / 255.0
    if p_norm.ndim == 2:
        p_norm = np.expand_dims(p_norm, 0)
    patch_tensors.append(torch.from_numpy(p_norm))

batch_tensor = torch.stack(patch_tensors, dim=0) # [21, 1, 384, 384]
print(f"    Input Batch Tensor: shape={batch_tensor.shape}, dtype={batch_tensor.dtype}", flush=True)

# 5. Inference
print("[5] Running model inference (torch.inference_mode())...", flush=True)
with torch.inference_mode():
    pred_fused, aux_outputs = model(batch_tensor)
    raw_nan = int(torch.isnan(pred_fused).sum().item())
    raw_inf = int(torch.isinf(pred_fused).sum().item())
    print(f"    Raw Fused Output Logits: shape={pred_fused.shape}, NaN={raw_nan}, Inf={raw_inf}", flush=True)
    assert raw_nan == 0 and raw_inf == 0, "Non-finite raw logits detected!"
    
    pred_probs = torch.sigmoid(pred_fused).squeeze(1).cpu().numpy() # [21, 384, 384]

recon_prob = reconstruct_panoramic(list(pred_probs), coords, full_shape=(H, W), patch_size=384)

prob_nan = int(np.isnan(recon_prob).sum())
prob_inf = int(np.isinf(recon_prob).sum())
prob_min = float(recon_prob.min())
prob_max = float(recon_prob.max())
prob_mean = float(recon_prob.mean())

print(f"    Reconstructed Probability Map: shape={recon_prob.shape}, min={prob_min:.6f}, max={prob_max:.6f}, mean={prob_mean:.6f}", flush=True)
print(f"    Probability Map Health: NaN count={prob_nan}, Inf count={prob_inf}", flush=True)
assert prob_nan == 0 and prob_inf == 0, "Non-finite probability values detected!"
assert 0.0 <= prob_min <= prob_max <= 1.0, "Probability values outside [0, 1] range!"

# 6. Apply tau = 0.50 Threshold
print("[6] Applying Operating Threshold tau = 0.50...", flush=True)
threshold = 0.50
binary_mask = (recon_prob >= threshold).astype(np.uint8)
pos_pixels = int(binary_mask.sum())
total_pixels = binary_mask.size
pos_pct = float(pos_pixels / total_pixels * 100)
is_zero_pred = bool(pos_pixels == 0)

print(f"    Binary Mask (tau={threshold}): Positive Pixels={pos_pixels}, Pct={pos_pct:.4f}%, ZeroPred={is_zero_pred}", flush=True)

# 7. Save Visualizations
print("[7] Generating temporary integration preflight visualization artifacts...", flush=True)
out_dir = base_dir / "outputs/diagnostics/FINAL_INTEGRATION_PREFLIGHT"
out_dir.mkdir(parents=True, exist_ok=True)

# Save original image
Image.fromarray(img_np).save(out_dir / "representative_original.png")

# Save probability map as normalized grayscale image (0-255)
prob_vis = (recon_prob * 255.0).clip(0, 255).astype(np.uint8)
Image.fromarray(prob_vis).save(out_dir / "representative_probability.png")

# Save binary mask (0 or 255)
mask_vis = (binary_mask * 255).astype(np.uint8)
Image.fromarray(mask_vis).save(out_dir / "representative_mask.png")

# Create Overlay (RGB with red mask overlay on grayscale image)
img_rgb = np.stack([img_np, img_np, img_np], axis=-1).astype(np.uint8)
overlay = img_rgb.copy()
mask_idx = binary_mask == 1
# Blend red [255, 40, 40] with 60% opacity over the original radiograph
overlay[mask_idx, 0] = np.clip(0.4 * overlay[mask_idx, 0] + 0.6 * 255, 0, 255).astype(np.uint8)
overlay[mask_idx, 1] = np.clip(0.4 * overlay[mask_idx, 1] + 0.6 * 40, 0, 255).astype(np.uint8)
overlay[mask_idx, 2] = np.clip(0.4 * overlay[mask_idx, 2] + 0.6 * 40, 0, 255).astype(np.uint8)

Image.fromarray(overlay).save(out_dir / "representative_overlay.png")
print("    Saved 4 visualization files to outputs/diagnostics/FINAL_INTEGRATION_PREFLIGHT/ successfully.", flush=True)

# 8. Save Metrics JSON for verification
preflight_summary = {
    "experiment": "EXP-MLUA-003",
    "checkpoint": "outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E56_FINAL.pth",
    "epoch": epoch,
    "global_step": step,
    "parameter_nan_count": nan_params,
    "parameter_inf_count": inf_params,
    "representative_image": "data/raw/DC1000_dataset/org_train_dataset/images/101.png",
    "input_raw_shape": [orig_h, orig_w],
    "input_panoramic_shape": [H, W],
    "patch_count": len(patches),
    "patch_size": 384,
    "stride": 192,
    "reconstructed_shape": list(recon_prob.shape),
    "probability_min": prob_min,
    "probability_max": prob_max,
    "probability_mean": prob_mean,
    "probability_nan_count": prob_nan,
    "probability_inf_count": prob_inf,
    "threshold": threshold,
    "binary_mask_positive_pixels": pos_pixels,
    "binary_mask_positive_percentage": pos_pct,
    "zero_prediction": is_zero_pred,
    "preflight_status": "PASS"
}

import json
with open(out_dir / "preflight_metrics.json", "w") as f:
    json.dump(preflight_summary, f, indent=2)

print("\nPREFLIGHT EXECUTION COMPLETED SUCCESSFULLY: ALL CHECKS PASSED.", flush=True)
