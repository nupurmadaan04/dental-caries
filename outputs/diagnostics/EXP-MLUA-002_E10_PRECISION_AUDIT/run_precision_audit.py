"""
EXP-MLUA-002: PRECISION & DTYPE FORENSIC AUDIT
Detailed Diagnostic Script to trace precision, dtypes, autocast contexts,
and internal mathematical operations of GroupNorm on Batch 68.
Strictly isolated & read-only.
"""

import sys
import math
import platform
from pathlib import Path

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

audit_dir = base_dir / "outputs" / "diagnostics" / "EXP-MLUA-002_E10_PRECISION_AUDIT"
log_file = audit_dir / "precision_audit_log.txt"


def log_msg(msg: str):
    print(msg, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run_audit():
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("EXP-MLUA-002: PRECISION & DTYPE FORENSIC AUDIT\n" + "=" * 80 + "\n")

    log_msg("=" * 80)
    log_msg("1. GLOBAL SYSTEM & PYTORCH PRECISION CONFIGURATION AUDIT")
    log_msg("=" * 80)

    log_msg(f"PyTorch Version: {torch.__version__}")
    log_msg(f"Python Version: {platform.python_version()} ({platform.system()} {platform.release()})")
    log_msg(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log_msg(f"CUDA Version: {torch.version.cuda}")
        log_msg(f"cuDNN Version: {torch.backends.cudnn.version()}")
        log_msg(f"GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        log_msg("Compute Hardware: CPU (8 OpenMP/MKL Worker Threads)")
    
    log_msg(f"Default Tensor Dtype: {torch.get_default_dtype()}")
    log_msg(f"Autocast GPU Enabled: {torch.is_autocast_enabled()}")
    log_msg(f"Autocast CPU Enabled: {torch.is_autocast_cpu_enabled() if hasattr(torch, 'is_autocast_cpu_enabled') else 'N/A'}")
    if hasattr(torch, "get_autocast_gpu_dtype"):
        try:
            log_msg(f"Autocast GPU Dtype: {torch.get_autocast_gpu_dtype()}")
        except Exception as e:
            log_msg(f"Autocast GPU Dtype: N/A ({e})")
    if hasattr(torch, "get_autocast_cpu_dtype"):
        try:
            log_msg(f"Autocast CPU Dtype: {torch.get_autocast_cpu_dtype()}")
        except Exception as e:
            log_msg(f"Autocast CPU Dtype: N/A ({e})")

    log_msg(f"torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32 if hasattr(torch.backends.cuda, 'matmul') else 'N/A'}")
    log_msg(f"torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32 if hasattr(torch.backends, 'cudnn') else 'N/A'}")

    log_msg("\n" + "=" * 80)
    log_msg("2. CHECKPOINT DTYPE & FINITENESS AUDIT (EXP-MLUA-002_LATEST.pth)")
    log_msg("=" * 80)

    config_file = base_dir / "configs" / "experiments" / "EXP-MLUA-002.yaml"
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    latest_ckpt_path = base_dir / "outputs" / "experiments" / "EXP-MLUA-002" / "checkpoints" / "EXP-MLUA-002_LATEST.pth"
    device = torch.device("cpu")
    torch.set_num_threads(8)

    ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
    log_msg(f"Checkpoint Epoch: {ckpt['epoch']}, Global Step: {ckpt.get('global_step', 'N/A')}")

    stu_dtypes = set(p.dtype for p in ckpt["model_stu_state_dict"].values() if isinstance(p, torch.Tensor))
    tea_dtypes = set(p.dtype for p in ckpt["model_tea_state_dict"].values() if isinstance(p, torch.Tensor))
    log_msg(f"Student Parameter Dtypes in Checkpoint: {stu_dtypes}")
    log_msg(f"Teacher Parameter Dtypes in Checkpoint: {tea_dtypes}")

    opt_state = ckpt["optimizer_state_dict"]["state"]
    opt_dtypes = set()
    for s in opt_state.values():
        for k, v in s.items():
            if isinstance(v, torch.Tensor):
                opt_dtypes.add(v.dtype)
    log_msg(f"Optimizer State Tensor Dtypes in Checkpoint: {opt_dtypes}")

    all_stu_finite = all(torch.isfinite(p).all().item() for p in ckpt["model_stu_state_dict"].values() if isinstance(p, torch.Tensor))
    all_tea_finite = all(torch.isfinite(p).all().item() for p in ckpt["model_tea_state_dict"].values() if isinstance(p, torch.Tensor))
    log_msg(f"All Student Parameters Finite: {all_stu_finite}")
    log_msg(f"All Teacher Parameters Finite: {all_tea_finite}")

    log_msg("\n" + "=" * 80)
    log_msg("3. REPRODUCING EXACT BATCH 68 INFERENCE & DTYPE TRACING")
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

    batch_size = cfg["data"]["batch_size"]
    l_batch_size = cfg["data"]["labeled_batch_size"]
    batch_sampler = TwoStreamBatchSampler(
        labeled_indices,
        unlabeled_indices,
        batch_size=batch_size,
        l_batch_size=l_batch_size,
    )

    train_dataset = CachedTrainDataset(train_img_files, train_lbl_files, transize=cfg["data"]["patch_size"])
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=0)

    model_stu = Net(in_c=1, out_c=1, encoder_weights=None).to(device)
    model_tea = Net(in_c=1, out_c=1, encoder_weights=None).to(device)
    for p in model_tea.parameters():
        p.requires_grad = False

    model_stu.load_state_dict(ckpt["model_stu_state_dict"])
    model_tea.load_state_dict(ckpt["model_tea_state_dict"])

    optimizer = torch.optim.AdamW(model_stu.parameters(), lr=cfg["optimization"]["learning_rate"], weight_decay=cfg["optimization"]["weight_decay"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    theta = cfg["ssl"]["ema_theta"]
    mc_t = cfg["ssl"]["mc_iterations"]
    noise_sigma = cfg["ssl"]["noise_sigma"]
    noise_clamp = cfg["ssl"]["noise_clamp"]
    transize = cfg["data"]["patch_size"]
    epoch = 9

    model_stu.train()
    model_tea.eval()

    log_msg("[Execution] Stepping through batches 0 to 67 to reach exact Batch 68 state...")
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

        # WE HAVE REACHED BATCH 68
        log_msg(f"\n[BATCH 68 REACHED] Global Step = {glob_step}")
        log_msg(f"Student Model Training Mode: {model_stu.training}")
        log_msg(f"Teacher Model Training Mode: {model_tea.training}")

        ul_imgs = imgs[l_batch_size:]
        repeated_ul = ul_imgs.repeat_interleave(mc_t + 1, dim=0)
        noise = torch.clamp(torch.randn_like(repeated_ul) * noise_sigma, -noise_clamp, noise_clamp)
        is_orig_mask = (torch.arange(repeated_ul.size(0), device=device) % (mc_t + 1) == 0).view(-1, 1, 1, 1)
        pert_ul_all = torch.where(is_orig_mask, repeated_ul, repeated_ul + noise)

        log_msg(f"\n[Dtype Audit along Teacher Forward Path on Batch 68]:")
        log_msg(f"  Input 'imgs' dtype: {imgs.dtype}")
        log_msg(f"  Input 'gts' dtype: {gts.dtype}")
        log_msg(f"  'ul_imgs' dtype: {ul_imgs.dtype}")
        log_msg(f"  'noise' dtype: {noise.dtype}")
        log_msg(f"  'pert_ul_all' dtype: {pert_ul_all.dtype}")

        # Execute inside torch.inference_mode as in production code
        with torch.inference_mode():
            log_msg(f"  Context torch.is_autocast_enabled(): {torch.is_autocast_enabled()}")
            if hasattr(torch, "is_autocast_cpu_enabled"):
                log_msg(f"  Context torch.is_autocast_cpu_enabled(): {torch.is_autocast_cpu_enabled()}")

            # 1. Teacher Encoder
            features = model_tea.encoder(pert_ul_all)
            for f_i, feat in enumerate(features):
                log_msg(f"  Teacher Encoder c{f_i}: shape={list(feat.shape)}, dtype={feat.dtype}, is_finite={torch.isfinite(feat).all().item()}")

            # 2. FPN top-down
            c2, c3, c4, c5 = features[-4:]
            p5 = model_tea.decoder.p5(c5)
            log_msg(f"  FPN p5: shape={list(p5.shape)}, dtype={p5.dtype}, is_finite={torch.isfinite(p5).all().item()}")

            # 3. SegmentationBlock[0] on p5
            seg_block_0 = model_tea.decoder.seg_blocks[0]
            conv3x3_gn_relu = seg_block_0.block[0]
            conv_op = conv3x3_gn_relu.block[0]
            gn_op = conv3x3_gn_relu.block[1]
            relu_op = conv3x3_gn_relu.block[2]

            log_msg(f"  GroupNorm Module Type: {type(gn_op)}")
            log_msg(f"  GroupNorm training mode: {gn_op.training}")
            log_msg(f"  GroupNorm num_groups: {gn_op.num_groups}")
            log_msg(f"  GroupNorm num_channels: {gn_op.num_channels}")
            log_msg(f"  GroupNorm eps: {gn_op.eps}")
            log_msg(f"  GroupNorm affine: {gn_op.affine}")
            log_msg(f"  GroupNorm weight dtype: {gn_op.weight.dtype}, min={gn_op.weight.min():.4f}, max={gn_op.weight.max():.4f}")
            log_msg(f"  GroupNorm bias dtype: {gn_op.bias.dtype}, min={gn_op.bias.min():.4f}, max={gn_op.bias.max():.4f}")

            conv_out = conv_op(p5)
            log_msg(f"  conv_out (GN input): shape={list(conv_out.shape)}, dtype={conv_out.dtype}, is_finite={torch.isfinite(conv_out).all().item()}")

            log_msg("\n" + "=" * 80)
            log_msg("4. DETAILED GROUPNORM INTERNAL MATHEMATICAL DECOMPOSITION")
            log_msg("=" * 80)

            # Inspect each unlabeled image's 9 MC slices (image 0: 0-9, image 1: 9-18, image 2: 18-27, image 3: 27-36)
            for u_idx in range(4):
                slice_conv = conv_out[u_idx * 9 : (u_idx + 1) * 9] # [9, 128, 12, 12]
                log_msg(f"\nUnlabeled Image {u_idx} (9 MC Slices) Input to GroupNorm:")
                log_msg(f"  Shape: {list(slice_conv.shape)}, Dtype: {slice_conv.dtype}")
                log_msg(f"  Min: {slice_conv.min().item():.6e}, Max: {slice_conv.max().item():.6e}")
                log_msg(f"  Mean: {slice_conv.mean().item():.6e}, Std: {slice_conv.std().item():.6e}")
                log_msg(f"  Is Finite: {torch.isfinite(slice_conv).all().item()}")

                # Manual step-by-step GroupNorm evaluation
                N, C, H, W = slice_conv.shape
                G = gn_op.num_groups # 32
                x_reshaped = slice_conv.view(N, G, C // G, H, W) # [9, 32, 4, 12, 12]
                
                # In FP32
                mean_fp32 = x_reshaped.mean(dim=[2, 3, 4], keepdim=True)
                var_fp32 = x_reshaped.var(dim=[2, 3, 4], keepdim=True, unbiased=False)
                var_plus_eps_fp32 = var_fp32 + gn_op.eps
                rstd_fp32 = 1.0 / torch.sqrt(var_plus_eps_fp32)

                log_msg(f"  [FP32 Manual] Mean min={mean_fp32.min():.6e}, max={mean_fp32.max():.6e}, finite={torch.isfinite(mean_fp32).all().item()}")
                log_msg(f"  [FP32 Manual] Var min={var_fp32.min():.6e}, max={var_fp32.max():.6e}, finite={torch.isfinite(var_fp32).all().item()}")
                log_msg(f"  [FP32 Manual] Var+eps min={var_plus_eps_fp32.min():.6e}, max={var_plus_eps_fp32.max():.6e}, finite={torch.isfinite(var_plus_eps_fp32).all().item()}")
                log_msg(f"  [FP32 Manual] Reciprocal Std min={rstd_fp32.min():.6e}, max={rstd_fp32.max():.6e}, finite={torch.isfinite(rstd_fp32).all().item()}")

                # PyTorch native forward
                gn_out_slice = gn_op(slice_conv)
                log_msg(f"  [PyTorch Native GN Output] Dtype={gn_out_slice.dtype}, Finite={torch.isfinite(gn_out_slice).all().item()}, NaNs={torch.isnan(gn_out_slice).sum().item()}")

                # Comparison with FP64
                slice_conv_fp64 = slice_conv.double()
                x_reshaped_fp64 = slice_conv_fp64.view(N, G, C // G, H, W)
                mean_fp64 = x_reshaped_fp64.mean(dim=[2, 3, 4], keepdim=True)
                var_fp64 = x_reshaped_fp64.var(dim=[2, 3, 4], keepdim=True, unbiased=False)
                gn_weight_fp64 = gn_op.weight.double().view(1, C, 1, 1)
                gn_bias_fp64 = gn_op.bias.double().view(1, C, 1, 1)
                mean_expanded_fp64 = mean_fp64.expand_as(x_reshaped_fp64).reshape(N, C, H, W)
                var_expanded_fp64 = var_fp64.expand_as(x_reshaped_fp64).reshape(N, C, H, W)
                norm_fp64 = (slice_conv_fp64 - mean_expanded_fp64) / torch.sqrt(var_expanded_fp64 + gn_op.eps)
                out_fp64 = norm_fp64 * gn_weight_fp64 + gn_bias_fp64

                log_msg(f"  [FP64 Diagnostic] Output Finite: {torch.isfinite(out_fp64).all().item()}, NaNs={torch.isnan(out_fp64).sum().item()}")

        break

    log_msg("\n" + "=" * 80)
    log_msg("AUDIT COMPLETE")
    log_msg("=" * 80)


if __name__ == "__main__":
    run_audit()
