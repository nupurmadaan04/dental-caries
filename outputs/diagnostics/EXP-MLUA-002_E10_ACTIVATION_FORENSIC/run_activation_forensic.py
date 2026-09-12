"""
EXP-MLUA-002: ACTIVATION & PARAMETER GROWTH FORENSIC SCRIPT
Read-Only Root-Cause Analysis of Feature Amplification in Teacher FPN.
Strictly isolated from EXP-MLUA-002 production environment.
"""

import os
import sys
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

base_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(base_dir))

from src.mlua.models.fpn import Net
from src.mlua.data.sampler import TwoStreamBatchSampler
from util.utils import DiceLoss, sigmoid_mse_loss, sigmoid_rampup, get_current_consistency_weight
from src.mlua.engine.train_exp002 import CachedTrainDataset, seed_everything

diag_dir = base_dir / "outputs" / "diagnostics" / "EXP-MLUA-002_E10_ACTIVATION_FORENSIC"
diag_dir.mkdir(parents=True, exist_ok=True)
log_file = diag_dir / "activation_forensic_log.txt"
json_file = diag_dir / "activation_forensic_data.json"


def log_msg(msg: str):
    print(msg, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def tensor_stats(t: torch.Tensor, name: str = "") -> Dict[str, Any]:
    if t is None:
        return {"name": name, "shape": None, "finite": True}
    is_fin = torch.isfinite(t).all().item()
    nan_cnt = torch.isnan(t).sum().item()
    inf_cnt = torch.isinf(t).sum().item()
    
    # Compute stats on finite elements or raw if all finite
    if is_fin:
        t_float = t.float()
        t_min = t_float.min().item()
        t_max = t_float.max().item()
        t_mean = t_float.mean().item()
        t_std = t_float.std().item() if t.numel() > 1 else 0.0
        t_abs_max = t_float.abs().max().item()
    else:
        fin_mask = torch.isfinite(t)
        if fin_mask.any():
            t_fin = t[fin_mask].float()
            t_min = t_fin.min().item()
            t_max = t_fin.max().item()
            t_mean = t_fin.mean().item()
            t_std = t_fin.std().item() if t_fin.numel() > 1 else 0.0
            t_abs_max = t_fin.abs().max().item()
        else:
            t_min, t_max, t_mean, t_std, t_abs_max = float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

    return {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "finite": is_fin,
        "nan_count": nan_cnt,
        "inf_count": inf_cnt,
        "min": t_min,
        "max": t_max,
        "mean": t_mean,
        "std": t_std,
        "abs_max": t_abs_max,
    }


def main():
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("EXP-MLUA-002: ACTIVATION & PARAMETER GROWTH FORENSIC REPORT LOG\n" + "=" * 80 + "\n")

    log_msg("=" * 80)
    log_msg("EXP-MLUA-002: ACTIVATION & PARAMETER GROWTH FORENSIC AUDIT")
    log_msg("=" * 80)

    cfg_path = base_dir / "configs" / "experiments" / "EXP-MLUA-002.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    torch.set_num_threads(8)

    results: Dict[str, Any] = {}

    # 1. PARAMETER MAGNITUDE AUDIT ACROSS CHECKPOINTS
    log_msg("\n[SECTION 1: PARAMETER MAGNITUDE AUDIT]")
    checkpoints_dir = base_dir / "outputs" / "experiments" / "EXP-MLUA-002" / "checkpoints"
    available_ckpts = sorted(list(checkpoints_dir.glob("*.pth")))
    log_msg(f"Available Checkpoints in EXP-MLUA-002: {[c.name for c in available_ckpts]}")

    ckpt_param_records = {}
    for c_path in available_ckpts:
        c_name = c_path.name
        ckpt_data = torch.load(c_path, map_location=device, weights_only=False)
        epoch = ckpt_data.get("epoch", "N/A")
        step = ckpt_data.get("global_step", "N/A")
        log_msg(f"\n--- Checkpoint: {c_name} (Epoch {epoch}, Step {step}) ---")

        tea_state = ckpt_data.get("model_tea_state_dict", {})
        stu_state = ckpt_data.get("model_stu_state_dict", {})

        layer_records = []
        for p_name, param in tea_state.items():
            if isinstance(param, torch.Tensor) and param.is_floating_point():
                stats = tensor_stats(param, f"tea.{p_name}")
                layer_records.append(stats)

        # Sort by abs_max
        layer_records.sort(key=lambda x: x["abs_max"] if not math.isnan(x["abs_max"]) else -1, reverse=True)
        log_msg("Top 10 Largest Parameters in Teacher:")
        for r in layer_records[:10]:
            log_msg(f"  {r['name']:<50} shape={str(r['shape']):<20} abs_max={r['abs_max']:.6e} mean={r['mean']:.6e} std={r['std']:.6e}")

        ckpt_param_records[c_name] = {
            "epoch": epoch,
            "global_step": step,
            "top_10_largest_parameters": layer_records[:10],
            "all_parameters_finite": all(r["finite"] for r in layer_records),
        }

    results["parameter_audit"] = ckpt_param_records

    # 2. RUN WARMUP TO BATCH 68 FOR DETAILED ACTIVATION TRACE
    log_msg("\n" + "=" * 80)
    log_msg("[SECTION 2: REPRODUCING BATCH 68 & DETAILED ACTIVATION TRACE]")
    log_msg("=" * 80)

    seed_everything(cfg["experiment"]["seed"])

    train_img_dir = base_dir / cfg["data"]["train_image_dir"]
    train_lbl_dir = base_dir / cfg["data"]["train_label_dir"]
    train_img_files = sorted(list(train_img_dir.glob("*.png")), key=lambda x: int(x.stem))
    train_lbl_files = sorted(list(train_lbl_dir.glob("*.png")), key=lambda x: int(x.stem))

    labeled_count = cfg["data"]["labeled_count"]
    all_indices = list(range(len(train_img_files)))
    labeled_indices = all_indices[:labeled_count]
    unlabeled_indices = all_indices[labeled_count:]

    batch_sampler = TwoStreamBatchSampler(
        labeled_indices, unlabeled_indices,
        batch_size=cfg["data"]["batch_size"],
        l_batch_size=cfg["data"]["labeled_batch_size"]
    )
    train_dataset = CachedTrainDataset(train_img_files, train_lbl_files, transize=cfg["data"]["patch_size"])
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=0)

    model_stu = Net(in_c=1, out_c=1, encoder_weights=None).to(device)
    model_tea = Net(in_c=1, out_c=1, encoder_weights=None).to(device)
    for p in model_tea.parameters():
        p.requires_grad = False

    latest_ckpt_path = checkpoints_dir / "EXP-MLUA-002_LATEST.pth"
    latest_ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
    model_stu.load_state_dict(latest_ckpt["model_stu_state_dict"])
    model_tea.load_state_dict(latest_ckpt["model_tea_state_dict"])

    optimizer = torch.optim.AdamW(model_stu.parameters(), lr=cfg["optimization"]["learning_rate"], weight_decay=cfg["optimization"]["weight_decay"])
    optimizer.load_state_dict(latest_ckpt["optimizer_state_dict"])

    theta = cfg["ssl"]["ema_theta"]
    mc_t = cfg["ssl"]["mc_iterations"]
    noise_sigma = cfg["ssl"]["noise_sigma"]
    noise_clamp = cfg["ssl"]["noise_clamp"]
    transize = cfg["data"]["patch_size"]
    l_batch_size = cfg["data"]["labeled_batch_size"]
    epoch = 9

    model_stu.train()
    model_tea.eval()

    log_msg("[Fast-Forward] Advancing through Batches 0 to 67...")
    for b_idx, (imgs, gts) in enumerate(train_loader):
        glob_step = epoch * len(train_loader) + b_idx + 1
        imgs = imgs.to(device)
        gts = gts.to(device)

        if b_idx < 68:
            pred_fused, pred_aux_list = model_stu(imgs)
            ul_imgs = imgs[l_batch_size:]
            repeated_ul = ul_imgs.repeat_interleave(mc_t + 1, dim=0)
            noise = torch.clamp(torch.randn_like(repeated_ul) * noise_sigma, -noise_clamp, noise_clamp)
            is_orig_mask = (torch.arange(repeated_ul.size(0), device=device) % (mc_t + 1) == 0).view(-1, 1, 1, 1)
            pert_ul_all = torch.where(is_orig_mask, repeated_ul, repeated_ul + noise)

            with torch.inference_mode():
                all_fused_tea, _ = model_tea(pert_ul_all)
                orig_indices = torch.arange(0, repeated_ul.size(0), mc_t + 1, device=device)
                ul_pred_tea = all_fused_tea[orig_indices]
                all_9 = all_fused_tea.view(l_batch_size, mc_t + 1, 1, transize, transize)
                mean_preds = torch.mean(all_9, dim=1).sigmoid()
                uncertainty = -2.0 * torch.sum(mean_preds * torch.log(mean_preds + 1e-6), dim=1, keepdim=True)

            threshold = (cfg["ssl"]["threshold_start_factor"] + (cfg["ssl"]["threshold_end_factor"] - cfg["ssl"]["threshold_start_factor"]) * sigmoid_rampup(glob_step, cfg["ssl"]["threshold_rampup_steps"])) * math.log(2)
            mask = (uncertainty < threshold).float()
            consistency_dist = sigmoid_mse_loss(pred_fused[l_batch_size:], ul_pred_tea)
            mask_sum = 2.0 * torch.sum(mask) + 1e-16
            consistency_loss = torch.sum(mask * consistency_dist) / mask_sum

            bce_loss = 0.0
            dice_loss = 0.0
            for pred_aux in pred_aux_list:
                bce_loss += F.binary_cross_entropy_with_logits(pred_aux[:l_batch_size], gts[:l_batch_size])
                dice_loss += DiceLoss()(pred_aux[:l_batch_size], gts[:l_batch_size])
            bce_loss += F.binary_cross_entropy_with_logits(pred_fused[:l_batch_size], gts[:l_batch_size])
            dice_loss += DiceLoss()(pred_fused[:l_batch_size], gts[:l_batch_size])
            seg_loss = 0.5 * (bce_loss / 4.0 + dice_loss / 4.0)

            consistency_weight = get_current_consistency_weight(epoch, cfg["ssl"]["consistency_rampup_epochs"], weight=cfg["ssl"]["consistency_weight_max"])
            total_loss = seg_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            alpha = min(1.0 - 1.0 / (epoch + 1), theta)
            for p_tea, p_stu in zip(model_tea.parameters(), model_stu.parameters()):
                p_tea.data.mul_(alpha).add_(p_stu.data, alpha=1.0 - alpha)
            continue

        # BATCH 68 REACHED
        log_msg(f"[BATCH 68 REACHED] Global Step = {glob_step}")
        ul_imgs = imgs[l_batch_size:] # [4, 1, 384, 384]
        repeated_ul = ul_imgs.repeat_interleave(mc_t + 1, dim=0)
        noise = torch.clamp(torch.randn_like(repeated_ul) * noise_sigma, -noise_clamp, noise_clamp)
        is_orig_mask = (torch.arange(repeated_ul.size(0), device=device) % (mc_t + 1) == 0).view(-1, 1, 1, 1)
        pert_ul_all = torch.where(is_orig_mask, repeated_ul, repeated_ul + noise)

        # 3. LAYER-BY-LAYER ACTIVATION COMPARISON (IMAGE 0 vs IMAGE 3)
        log_msg("\n" + "=" * 80)
        log_msg("[SECTION 3: LAYER-BY-LAYER ACTIVATION TRACE: NORMAL (IMG 0) VS FAILING (IMG 3)]")
        log_msg("=" * 80)

        # We will trace Image 0 (slice 0:9) and Image 3 (slice 27:36)
        img0_pert = pert_ul_all[0:9] # Normal
        img3_pert = pert_ul_all[27:36] # Failing

        def trace_sample_activations(sample_tensor: torch.Tensor, label: str) -> List[Dict[str, Any]]:
            trace = []
            with torch.inference_mode():
                # Input
                trace.append(tensor_stats(sample_tensor, "00_Input"))

                # Encoder
                c0 = sample_tensor
                x = model_tea.encoder.conv1(c0)
                trace.append(tensor_stats(x, "01_encoder.conv1"))
                x = model_tea.encoder.bn1(x)
                trace.append(tensor_stats(x, "02_encoder.bn1"))
                c1 = model_tea.encoder.relu(x)
                trace.append(tensor_stats(c1, "03_encoder.relu (c1)"))
                x = model_tea.encoder.maxpool(c1)
                trace.append(tensor_stats(x, "04_encoder.maxpool"))
                c2 = model_tea.encoder.layer1(x)
                trace.append(tensor_stats(c2, "05_encoder.layer1 (c2)"))
                c3 = model_tea.encoder.layer2(c2)
                trace.append(tensor_stats(c3, "06_encoder.layer2 (c3)"))
                c4 = model_tea.encoder.layer3(c3)
                trace.append(tensor_stats(c4, "07_encoder.layer3 (c4)"))
                c5 = model_tea.encoder.layer4(c4)
                trace.append(tensor_stats(c5, "08_encoder.layer4 (c5)"))

                # FPN Top-Down
                p5 = model_tea.decoder.p5(c5)
                trace.append(tensor_stats(p5, "09_decoder.p5"))

                # p4 = p5_upsample + c4_skip
                p5_up = F.interpolate(p5, scale_factor=2, mode="nearest")
                trace.append(tensor_stats(p5_up, "10_p5_upsampled_to_p4"))
                c4_skip = model_tea.decoder.p4.skip_conv(c4)
                trace.append(tensor_stats(c4_skip, "11_p4.skip_conv(c4)"))
                p4 = p5_up + c4_skip
                trace.append(tensor_stats(p4, "12_decoder.p4 (sum)"))

                # p3 = p4_upsample + c3_skip
                p4_up = F.interpolate(p4, scale_factor=2, mode="nearest")
                c3_skip = model_tea.decoder.p3.skip_conv(c3)
                p3 = p4_up + c3_skip
                trace.append(tensor_stats(p3, "13_decoder.p3 (sum)"))

                # p2 = p3_upsample + c2_skip
                p3_up = F.interpolate(p3, scale_factor=2, mode="nearest")
                c2_skip = model_tea.decoder.p2.skip_conv(c2)
                p2 = p3_up + c2_skip
                trace.append(tensor_stats(p2, "14_decoder.p2 (sum)"))

                # Segmentation Block 0 (on p5)
                seg0_conv = model_tea.decoder.seg_blocks[0].block[0].block[0](p5)
                trace.append(tensor_stats(seg0_conv, "15_seg_blocks[0].conv3x3(p5)"))
                seg0_gn = model_tea.decoder.seg_blocks[0].block[0].block[1](seg0_conv)
                trace.append(tensor_stats(seg0_gn, "16_seg_blocks[0].gn(conv)"))
                seg0_relu = model_tea.decoder.seg_blocks[0].block[0].block[2](seg0_gn)
                trace.append(tensor_stats(seg0_relu, "17_seg_blocks[0].relu"))

                # Entire seg_blocks outputs
                fp_p5 = model_tea.decoder.seg_blocks[0](p5)
                trace.append(tensor_stats(fp_p5, "18_feature_pyramid[0] (p5_branch)"))
                fp_p4 = model_tea.decoder.seg_blocks[1](p4)
                trace.append(tensor_stats(fp_p4, "19_feature_pyramid[1] (p4_branch)"))
                fp_p3 = model_tea.decoder.seg_blocks[2](p3)
                trace.append(tensor_stats(fp_p3, "20_feature_pyramid[2] (p3_branch)"))
                fp_p2 = model_tea.decoder.seg_blocks[3](p2)
                trace.append(tensor_stats(fp_p2, "21_feature_pyramid[3] (p2_branch)"))

                # Merge
                merged = model_tea.decoder.merge([fp_p5, fp_p4, fp_p3, fp_p2])
                trace.append(tensor_stats(merged, "22_decoder.merge (sum)"))

                # Segmentation Head
                head_out = model_tea.segmentation_head(merged)
                trace.append(tensor_stats(head_out, "23_segmentation_head (fused)"))

            return trace

        trace_normal = trace_sample_activations(img0_pert, "Normal (Image 0)")
        trace_failing = trace_sample_activations(img3_pert, "Failing (Image 3)")

        log_msg(f"\n{'Layer':<35} | {'Normal abs.max':<16} | {'Failing abs.max':<16} | {'Normal std':<14} | {'Failing std':<14} | {'Status':<10}")
        log_msg("-" * 115)

        comparison_table = []
        first_abnormal_layer = None
        for r_norm, r_fail in zip(trace_normal, trace_failing):
            name = r_norm["name"]
            norm_max = r_norm["abs_max"]
            fail_max = r_fail["abs_max"]
            norm_std = r_norm["std"]
            fail_std = r_fail["std"]
            status = "FINITE" if r_fail["finite"] else "NON-FINITE"
            
            norm_max_str = f"{norm_max:.4e}" if not math.isnan(norm_max) else "NaN"
            fail_max_str = f"{fail_max:.4e}" if not math.isnan(fail_max) else "NaN"
            norm_std_str = f"{norm_std:.4e}" if not math.isnan(norm_std) else "NaN"
            fail_std_str = f"{fail_std:.4e}" if not math.isnan(fail_std) else "NaN"

            log_msg(f"{name:<35} | {norm_max_str:<16} | {fail_max_str:<16} | {norm_std_str:<14} | {fail_std_str:<14} | {status:<10}")
            
            comparison_table.append({
                "layer": name,
                "normal_abs_max": norm_max,
                "failing_abs_max": fail_max,
                "normal_std": norm_std,
                "failing_std": fail_std,
                "status": status,
            })

            if not r_fail["finite"] and first_abnormal_layer is None:
                first_abnormal_layer = name

        results["activation_comparison"] = comparison_table
        results["first_abnormal_layer"] = first_abnormal_layer

        # 4. DETAILED MATHEMATICAL GROUPNORM & FP32 OVERFLOW DECOMPOSITION
        log_msg("\n" + "=" * 80)
        log_msg("[SECTION 4: GROUPNORM MATHEMATICAL SCALE & OVERFLOW DECOMPOSITION]")
        log_msg("=" * 80)

        # Inspect Image 3 input to seg0_conv
        with torch.inference_mode():
            c5_img3 = model_tea.encoder.layer4(model_tea.encoder.layer3(model_tea.encoder.layer2(model_tea.encoder.layer1(model_tea.encoder.maxpool(model_tea.encoder.relu(model_tea.encoder.bn1(model_tea.encoder.conv1(img3_pert))))))))
            p5_img3 = model_tea.decoder.p5(c5_img3)
            conv_out_img3 = model_tea.decoder.seg_blocks[0].block[0].block[0](p5_img3)

            # Evaluate in FP64
            x_fp64 = conv_out_img3.double()
            N, C, H, W = x_fp64.shape
            G = 32
            x_grouped = x_fp64.view(N, G, C // G, H, W)
            mean_fp64 = x_grouped.mean(dim=[2, 3, 4], keepdim=True)
            centered_fp64 = x_grouped - mean_fp64
            sq_dev_fp64 = centered_fp64 ** 2
            sum_sq_dev_fp64 = sq_dev_fp64.sum(dim=[2, 3, 4], keepdim=True)
            var_fp64 = x_grouped.var(dim=[2, 3, 4], keepdim=True, unbiased=False)

            max_centered = centered_fp64.abs().max().item()
            max_sum_sq_dev = sum_sq_dev_fp64.max().item()
            flt_max = np.finfo(np.float32).max  # 3.4028235e+38
            ratio = max_sum_sq_dev / flt_max

            log_msg(f"Max |x - mean| in FP64: {max_centered:.6e}")
            log_msg(f"Max sum((x - mean)^2) per group in FP64: {max_sum_sq_dev:.6e}")
            log_msg(f"IEEE 754 Float32 FLT_MAX: {flt_max:.6e}")
            log_msg(f"Ratio (sum_sq_dev / FLT_MAX): {ratio:.6e} ({'OVERFLOW RISK > 1.0' if ratio > 1.0 else 'Safe <= 1.0'})")

            results["groupnorm_overflow_metrics"] = {
                "max_centered_abs": max_centered,
                "max_sum_sq_dev_fp64": max_sum_sq_dev,
                "flt_max_fp32": float(flt_max),
                "overflow_ratio": float(ratio),
                "overflow_confirmed": bool(ratio > 1.0),
            }

        break

    # Save JSON results
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log_msg(f"\n[Saved] Forensic structured data written to {json_file}")


if __name__ == "__main__":
    main()
