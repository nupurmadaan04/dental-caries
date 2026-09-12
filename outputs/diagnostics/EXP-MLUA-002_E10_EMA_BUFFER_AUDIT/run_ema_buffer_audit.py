import os
import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(base_dir))

import hashlib
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.mlua.models.fpn import Net
from src.mlua.data.sampler import TwoStreamBatchSampler
from src.mlua.engine.train_exp002 import CachedTrainDataset, seed_everything

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== STARTING EXP-MLUA-002 TEACHER EMA BUFFER AUDIT ===", flush=True)
    seed_everything(42)
    device = torch.device("cpu")

    cfg_path = base_dir / "configs" / "experiments" / "EXP-MLUA-002.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ckpt_latest_path = base_dir / "outputs" / "experiments" / "EXP-MLUA-002" / "checkpoints" / "EXP-MLUA-002_LATEST.pth"
    ckpt_best_path = base_dir / "outputs" / "experiments" / "EXP-MLUA-002" / "checkpoints" / "EXP-MLUA-002_BEST.pth"

    latest_sha = compute_sha256(ckpt_latest_path)
    best_sha = compute_sha256(ckpt_best_path)
    print(f"Latest Checkpoint SHA256: {latest_sha}")
    print(f"Best Checkpoint SHA256:   {best_sha}")

    checkpoint = torch.load(ckpt_latest_path, map_location=device, weights_only=False)
    epoch_saved = checkpoint.get("epoch", None)
    global_step_saved = checkpoint.get("global_step", None)
    print(f"Checkpoint Epoch: {epoch_saved}, Global Step: {global_step_saved}")

    model_stu = Net(in_c=1, out_c=1, encoder_weights=None).to(device)
    model_tea = Net(in_c=1, out_c=1, encoder_weights=None).to(device)

    model_stu.load_state_dict(checkpoint["model_stu_state_dict"])
    model_tea.load_state_dict(checkpoint["model_tea_state_dict"])

    model_stu.eval()
    model_tea.eval()

    # 1. Teacher Buffer Inventory
    tea_buffers = dict(model_tea.named_buffers())
    stu_buffers = dict(model_stu.named_buffers())
    print(f"Total Teacher Buffers: {len(tea_buffers)}")
    print(f"Total Student Buffers: {len(stu_buffers)}")

    bn_running_mean_keys = [k for k in tea_buffers if k.endswith("running_mean")]
    bn_running_var_keys = [k for k in tea_buffers if k.endswith("running_var")]
    bn_num_batches_keys = [k for k in tea_buffers if k.endswith("num_batches_tracked")]
    other_buffer_keys = [k for k in tea_buffers if not (k.endswith("running_mean") or k.endswith("running_var") or k.endswith("num_batches_tracked"))]

    print(f"BatchNorm running_mean count: {len(bn_running_mean_keys)}")
    print(f"BatchNorm running_var count: {len(bn_running_var_keys)}")
    print(f"BatchNorm num_batches_tracked count: {len(bn_num_batches_keys)}")
    print(f"Other buffers count: {len(other_buffer_keys)}")

    # 2. Compare Student vs Teacher BatchNorm statistics
    bn_comparison = []
    default_mean_count = 0
    default_var_count = 0
    default_batches_count = 0

    for mean_key in bn_running_mean_keys:
        prefix = mean_key[:-len(".running_mean")]
        var_key = f"{prefix}.running_var"
        nbt_key = f"{prefix}.num_batches_tracked"

        s_mean = stu_buffers[mean_key].float()
        t_mean = tea_buffers[mean_key].float()
        s_var = stu_buffers[var_key].float()
        t_var = tea_buffers[var_key].float()
        s_nbt = stu_buffers[nbt_key].item()
        t_nbt = tea_buffers[nbt_key].item()

        t_mean_is_default = bool(torch.all(t_mean == 0.0).item())
        t_var_is_default = bool(torch.all(t_var == 1.0).item())
        t_nbt_is_default = (t_nbt == 0)

        if t_mean_is_default:
            default_mean_count += 1
        if t_var_is_default:
            default_var_count += 1
        if t_nbt_is_default:
            default_batches_count += 1

        bn_comparison.append({
            "layer": prefix,
            "stu_mean_range": [float(s_mean.min().item()), float(s_mean.max().item())],
            "tea_mean_range": [float(t_mean.min().item()), float(t_mean.max().item())],
            "stu_var_range": [float(s_var.min().item()), float(s_var.max().item())],
            "tea_var_range": [float(t_var.min().item()), float(t_var.max().item())],
            "stu_nbt": int(s_nbt),
            "tea_nbt": int(t_nbt),
            "tea_mean_default": t_mean_is_default,
            "tea_var_default": t_var_is_default,
            "tea_nbt_default": t_nbt_is_default
        })

    print(f"Teacher default running_mean: {default_mean_count}/{len(bn_running_mean_keys)} ({default_mean_count/len(bn_running_mean_keys)*100:.1f}%)")
    print(f"Teacher default running_var:  {default_var_count}/{len(bn_running_var_keys)} ({default_var_count/len(bn_running_var_keys)*100:.1f}%)")
    print(f"Teacher default num_batches:  {default_batches_count}/{len(bn_num_batches_keys)} ({default_batches_count/len(bn_num_batches_keys)*100:.1f}%)")

    # 3. Parameter EMA Verification
    tea_params = dict(model_tea.named_parameters())
    stu_params = dict(model_stu.named_parameters())
    param_checks = []
    for k in list(tea_params.keys())[:15]:
        tp = tea_params[k].data
        sp = stu_params[k].data
        diff = torch.abs(tp - sp).max().item()
        param_checks.append({
            "param": k,
            "shape": list(tp.shape),
            "tea_abs_max": float(tp.abs().max().item()),
            "stu_abs_max": float(sp.abs().max().item()),
            "max_abs_diff": float(diff),
            "tea_finite": bool(torch.isfinite(tp).all().item())
        })

    # 4. Reproduce failing batch data
    train_img_dir = base_dir / cfg["data"]["train_image_dir"]
    train_lbl_dir = base_dir / cfg["data"]["train_label_dir"]
    train_img_files = sorted(list(train_img_dir.glob("*.png")), key=lambda x: int(x.stem))
    train_lbl_files = sorted(list(train_lbl_dir.glob("*.png")), key=lambda x: int(x.stem))
    
    labeled_count = 530
    all_indices = list(range(len(train_img_files)))
    labeled_indices = all_indices[:labeled_count]
    unlabeled_indices = all_indices[labeled_count:]

    batch_sampler = TwoStreamBatchSampler(
        labeled_indices,
        unlabeled_indices,
        batch_size=8,
        l_batch_size=4,
    )
    train_dataset = CachedTrainDataset(train_img_files, train_lbl_files, transize=384)
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=0)

    # Step through batches up to batch 68
    data_iter = iter(train_loader)
    for b_idx in range(69):
        batch = next(data_iter)
        if b_idx == 68:
            imgs, gts = batch
            imgs = imgs.to(device)
            u_imgs = imgs[4:]
            break

    # MC perturbation for failing image (Image 3)
    mc_t = 8
    noise_sigma = 0.01
    noise_clamp = 0.1
    u_expanded = u_imgs.unsqueeze(1).repeat(1, mc_t + 1, 1, 1, 1) # [4, 9, 1, 384, 384]
    noise = torch.randn_like(u_expanded[:, 1:]) * noise_sigma
    noise = torch.clamp(noise, -noise_clamp, noise_clamp)
    u_expanded[:, 1:] = u_expanded[:, 1:] + noise
    u_perturbed_all = u_expanded.view(-1, 1, 384, 384) # [36, 1, 384, 384]

    # Failing input: Image 3 perturbations (indices 27 to 35 in u_perturbed_all)
    failing_input = u_perturbed_all[27:36] # [9, 1, 384, 384]
    normal_input = u_perturbed_all[0:9]    # [9, 1, 384, 384]

    # 5. Counterfactual Diagnostic Pipeline
    def evaluate_model_pipeline(model, x_in):
        stats = {}
        with torch.no_grad():
            x = x_in
            # conv1
            x = model.encoder.conv1(x)
            stats["conv1"] = float(x.abs().max().item())
            x = model.encoder.bn1(x)
            stats["bn1"] = float(x.abs().max().item())
            c1 = model.encoder.relu(x)
            stats["c1"] = float(c1.abs().max().item())
            
            x = model.encoder.maxpool(c1)
            c2 = model.encoder.layer1(x)
            stats["c2"] = float(c2.abs().max().item())
            c3 = model.encoder.layer2(c2)
            stats["c3"] = float(c3.abs().max().item())
            c4 = model.encoder.layer3(c3)
            stats["c4"] = float(c4.abs().max().item())
            c5 = model.encoder.layer4(c4)
            stats["c5"] = float(c5.abs().max().item())

            # FPN decoder
            p5 = model.decoder.p5(c5)
            stats["p5"] = float(p5.abs().max().item())
            p4 = model.decoder.p4(p5, c4)
            p3 = model.decoder.p3(p4, c3)
            p2 = model.decoder.p2(p3, c2)

            # Decoder seg_blocks[0]
            blk0 = model.decoder.seg_blocks[0].block[0].block
            conv_out = blk0[0](p5)
            stats["seg0_conv"] = float(conv_out.abs().max().item())
            gn_in_finite = bool(torch.isfinite(conv_out).all().item())
            
            gn_out = blk0[1](conv_out)
            gn_out_finite = bool(torch.isfinite(gn_out).all().item())
            stats["gn_in_finite"] = gn_in_finite
            stats["gn_out_finite"] = gn_out_finite
            stats["gn_out_abs_max"] = float(gn_out.abs().max().item()) if gn_out_finite else float("nan")

            try:
                pred_fused, pred_aux_list = model(x_in)
                fused_finite = bool(torch.isfinite(pred_fused).all().item())
                stats["fused_finite"] = fused_finite
                if fused_finite:
                    prob = torch.sigmoid(pred_fused)
                    stats["max_fg_prob"] = float(prob.max().item())
                else:
                    stats["max_fg_prob"] = float("nan")
            except Exception:
                stats["fused_finite"] = False
                stats["max_fg_prob"] = float("nan")

        return stats

    # Condition A: Real Teacher (as loaded)
    cond_a_stats = evaluate_model_pipeline(model_tea, failing_input)
    print("\n--- CONDITION A (ACTUAL TEACHER WITH DEFAULT BN BUFFERS) ---")
    for k, v in cond_a_stats.items():
        print(f"  {k}: {v}")

    # Condition B: Diagnostic In-Memory Copy with Student BN Buffers Synchronized
    model_tea_synced = Net(in_c=1, out_c=1, encoder_weights=None).to(device)
    model_tea_synced.load_state_dict(checkpoint["model_tea_state_dict"])
    with torch.no_grad():
        for b_name, b_val in model_stu.named_buffers():
            dict(model_tea_synced.named_buffers())[b_name].copy_(b_val)
    model_tea_synced.eval()

    cond_b_stats = evaluate_model_pipeline(model_tea_synced, failing_input)
    print("\n--- CONDITION B (COUNTERFACTUAL TEACHER WITH SYNCHRONIZED BN BUFFERS) ---")
    for k, v in cond_b_stats.items():
        print(f"  {k}: {v}")

    # Condition C: Student Model evaluation for comparison
    cond_stu_stats = evaluate_model_pipeline(model_stu, failing_input)
    print("\n--- STUDENT MODEL (FOR COMPARISON) ---")
    for k, v in cond_stu_stats.items():
        print(f"  {k}: {v}")

    # Check test set directory integrity (read-only count)
    test_img_dir = base_dir / "data" / "raw" / "DC1000_dataset" / "org_test_dataset" / "images"
    test_lbl_dir = base_dir / "data" / "raw" / "DC1000_dataset" / "org_test_dataset" / "labels"
    test_imgs = sorted(os.listdir(test_img_dir))
    test_lbls = sorted(os.listdir(test_lbl_dir))
    print(f"\nTest set image count: {len(test_imgs)}, label count: {len(test_lbls)}")

    results = {
        "checkpoint_integrity": {
            "latest_sha256": latest_sha,
            "best_sha256": best_sha,
            "epoch_saved": epoch_saved,
            "global_step_saved": global_step_saved,
            "test_set_cases": len(test_imgs)
        },
        "buffer_inventory": {
            "total_buffers": len(tea_buffers),
            "bn_layers": len(bn_running_mean_keys),
            "default_running_mean_count": default_mean_count,
            "default_running_var_count": default_var_count,
            "default_num_batches_count": default_batches_count
        },
        "bn_comparison": bn_comparison,
        "param_checks": param_checks,
        "condition_a_actual_teacher": cond_a_stats,
        "condition_b_synced_teacher": cond_b_stats,
        "condition_student": cond_stu_stats
    }

    out_dir = base_dir / "outputs" / "diagnostics" / "EXP-MLUA-002_E10_EMA_BUFFER_AUDIT"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ema_buffer_audit_data.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved diagnostic JSON successfully.")

if __name__ == "__main__":
    main()
