"""
EXP-MLUA-002: EPOCH 10 FORENSIC TRACER
Isolated Diagnostic Tool to pinpoint the exact first non-finite operation.
Strictly Read-Only: Does NOT modify any production files or checkpoints.
"""

import os
import sys
import time
import math
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
from src.mlua.engine.train_exp002 import CachedTrainDataset, DeterministicSubsetDataset, seed_everything

diag_dir = base_dir / "outputs" / "diagnostics" / "EXP-MLUA-002_E10_FORENSIC"
diag_dir.mkdir(parents=True, exist_ok=True)
log_file = diag_dir / "e10_trace_log.txt"


def log_msg(msg: str):
    print(msg, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def check_tensor(t: torch.Tensor, name: str) -> bool:
    """Returns True if finite, False if contains NaN/Inf."""
    if t is None:
        return True
    if not torch.isfinite(t).all():
        num_nan = torch.isnan(t).sum().item()
        num_inf = torch.isinf(t).sum().item()
        log_msg(f"  [NON-FINITE DETECTED] Tensor '{name}' shape={list(t.shape)} dtype={t.dtype} | NaNs={num_nan}, Infs={num_inf}")
        return False
    return True


def run_forensic_trace():
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("EXP-MLUA-002: EPOCH 10 FORENSIC BATCH-BY-BATCH TRACE\n" + "=" * 80 + "\n")

    log_msg("=" * 80)
    log_msg("EXP-MLUA-002: EPOCH 10 FORENSIC BATCH-BY-BATCH TRACE")
    log_msg("=" * 80)

    config_file = base_dir / "configs" / "experiments" / "EXP-MLUA-002.yaml"
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    latest_ckpt_path = base_dir / "outputs" / "experiments" / "EXP-MLUA-002" / "checkpoints" / "EXP-MLUA-002_LATEST.pth"
    device = torch.device("cpu")
    torch.set_num_threads(8)

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

    lr = cfg["optimization"]["learning_rate"]
    weight_decay = cfg["optimization"]["weight_decay"]
    optimizer = torch.optim.AdamW(model_stu.parameters(), lr=lr, weight_decay=weight_decay)

    max_epochs = cfg["experiment"]["max_epochs"]
    poly_lr_fn = lambda epoch: (1.0 - float(epoch) / max_epochs) ** cfg["optimization"]["poly_power"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr_fn)

    bce_loss_fn = F.binary_cross_entropy_with_logits
    dice_loss_fn = DiceLoss()

    theta = cfg["ssl"]["ema_theta"]
    mc_t = cfg["ssl"]["mc_iterations"]
    noise_sigma = cfg["ssl"]["noise_sigma"]
    noise_clamp = cfg["ssl"]["noise_clamp"]
    transize = cfg["data"]["patch_size"]

    log_msg(f"[Loading Checkpoint] {latest_ckpt_path}")
    ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
    completed_epoch = ckpt["epoch"]
    model_stu.load_state_dict(ckpt["model_stu_state_dict"])
    model_tea.load_state_dict(ckpt["model_tea_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    log_msg(f"[Checkpoint Loaded] Completed Epoch: {completed_epoch}, Step: {ckpt.get('global_step', 'N/A')}")
    assert completed_epoch == 9

    # Verify all checkpoint parameters are finite before starting
    for name, param in model_stu.named_parameters():
        if not check_tensor(param, f"stu_init.{name}"):
            raise RuntimeError("Initial student parameter is non-finite!")
    for name, param in model_tea.named_parameters():
        if not check_tensor(param, f"tea_init.{name}"):
            raise RuntimeError("Initial teacher parameter is non-finite!")

    epoch = 9  # 0-indexed for Epoch 10
    epoch_num = 10
    glob_step = epoch * len(train_loader)

    model_stu.train()
    model_tea.eval()

    log_msg(f"\n[Forensic Execution] Starting trace for Epoch {epoch_num} (132 batches)...")

    first_failure = None

    for b_idx, (imgs, gts) in enumerate(train_loader):
        glob_step += 1
        imgs = imgs.to(device)
        gts = gts.to(device)

        # 1. Inputs check
        if not check_tensor(imgs, f"Batch {b_idx}: imgs"):
            first_failure = ("Input", "imgs", b_idx, glob_step)
            break
        if not check_tensor(gts, f"Batch {b_idx}: gts"):
            first_failure = ("Input", "gts", b_idx, glob_step)
            break

        # 2. Student Forward Pass
        pred_fused, pred_aux_list = model_stu(imgs)
        if not check_tensor(pred_fused, f"Batch {b_idx}: pred_fused"):
            first_failure = ("Student Forward", "pred_fused", b_idx, glob_step)
            break
        for aux_i, pred_aux in enumerate(pred_aux_list):
            if not check_tensor(pred_aux, f"Batch {b_idx}: pred_aux[{aux_i}]"):
                first_failure = ("Student Forward", f"pred_aux[{aux_i}]", b_idx, glob_step)
                break
        if first_failure:
            break

        # 3. Teacher Forward Pass
        ul_imgs = imgs[l_batch_size:]
        repeated_ul = ul_imgs.repeat_interleave(mc_t + 1, dim=0)
        noise = torch.clamp(torch.randn_like(repeated_ul) * noise_sigma, -noise_clamp, noise_clamp)
        is_orig_mask = (torch.arange(repeated_ul.size(0), device=device) % (mc_t + 1) == 0).view(-1, 1, 1, 1)
        pert_ul_all = torch.where(is_orig_mask, repeated_ul, repeated_ul + noise)

        if not check_tensor(pert_ul_all, f"Batch {b_idx}: pert_ul_all"):
            first_failure = ("Teacher Input", "pert_ul_all", b_idx, glob_step)
            break

        with torch.inference_mode():
            all_fused_tea, _ = model_tea(pert_ul_all)
            if not check_tensor(all_fused_tea, f"Batch {b_idx}: all_fused_tea"):
                first_failure = ("Teacher Forward", "all_fused_tea", b_idx, glob_step)
                break

            orig_indices = torch.arange(0, repeated_ul.size(0), mc_t + 1, device=device)
            ul_pred_tea = all_fused_tea[orig_indices]

            all_9 = all_fused_tea.view(l_batch_size, mc_t + 1, 1, transize, transize)
            mean_preds = torch.mean(all_9, dim=1).sigmoid()
            if not check_tensor(mean_preds, f"Batch {b_idx}: mean_preds"):
                first_failure = ("MC Uncertainty", "mean_preds", b_idx, glob_step)
                break

            # Uncertainty calculation
            uncertainty = -2.0 * torch.sum(mean_preds * torch.log(mean_preds + 1e-6), dim=1, keepdim=True)
            if not check_tensor(uncertainty, f"Batch {b_idx}: uncertainty"):
                first_failure = ("MC Uncertainty", "uncertainty", b_idx, glob_step)
                break

        if first_failure:
            break

        # Threshold and mask
        threshold = (cfg["ssl"]["threshold_start_factor"] + (cfg["ssl"]["threshold_end_factor"] - cfg["ssl"]["threshold_start_factor"]) * sigmoid_rampup(glob_step, cfg["ssl"]["threshold_rampup_steps"])) * math.log(2)
        mask = (uncertainty < threshold).float()
        if not check_tensor(mask, f"Batch {b_idx}: mask"):
            first_failure = ("MC Mask", "mask", b_idx, glob_step)
            break

        # Consistency loss
        consistency_dist = sigmoid_mse_loss(pred_fused[l_batch_size:], ul_pred_tea)
        if not check_tensor(consistency_dist, f"Batch {b_idx}: consistency_dist"):
            first_failure = ("Consistency", "consistency_dist", b_idx, glob_step)
            break

        mask_sum = 2.0 * torch.sum(mask) + 1e-16
        consistency_loss = torch.sum(mask * consistency_dist) / mask_sum
        if not check_tensor(consistency_loss, f"Batch {b_idx}: consistency_loss"):
            first_failure = ("Consistency", "consistency_loss", b_idx, glob_step)
            break

        # Supervised loss
        bce_loss = 0.0
        dice_loss = 0.0
        for pred_aux in pred_aux_list:
            bce_loss += bce_loss_fn(pred_aux[:l_batch_size], gts[:l_batch_size])
            dice_loss += dice_loss_fn(pred_aux[:l_batch_size], gts[:l_batch_size])
        bce_loss += bce_loss_fn(pred_fused[:l_batch_size], gts[:l_batch_size])
        dice_loss += dice_loss_fn(pred_fused[:l_batch_size], gts[:l_batch_size])
        seg_loss = 0.5 * (bce_loss / 4.0 + dice_loss / 4.0)

        if not check_tensor(seg_loss, f"Batch {b_idx}: seg_loss"):
            first_failure = ("Supervised Loss", "seg_loss", b_idx, glob_step)
            break

        consistency_weight = get_current_consistency_weight(epoch, cfg["ssl"]["consistency_rampup_epochs"], weight=cfg["ssl"]["consistency_weight_max"])
        total_loss = seg_loss + consistency_weight * consistency_loss

        if not check_tensor(total_loss, f"Batch {b_idx}: total_loss"):
            first_failure = ("Total Loss", "total_loss", b_idx, glob_step)
            break

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Check gradients
        grad_nonfinite_param = None
        max_grad_norm = 0.0
        for name, p in model_stu.named_parameters():
            if p.grad is not None:
                g_norm = p.grad.norm().item()
                if g_norm > max_grad_norm:
                    max_grad_norm = g_norm
                if not torch.isfinite(p.grad).all():
                    grad_nonfinite_param = name
                    log_msg(f"  [GRAD NON-FINITE] Parameter '{name}' grad has NaNs/Infs: {p.grad.isnan().sum().item()} NaNs, {p.grad.isinf().sum().item()} Infs")
                    break

        if grad_nonfinite_param:
            first_failure = ("Backward Gradient", f"grad({grad_nonfinite_param})", b_idx, glob_step)
            break

        # Optimizer step
        optimizer.step()

        # Check student parameters after step
        param_nonfinite = None
        for name, p in model_stu.named_parameters():
            if not torch.isfinite(p).all():
                param_nonfinite = name
                log_msg(f"  [PARAM NON-FINITE AFTER STEP] Parameter '{name}' has NaNs/Infs after optimizer.step()")
                break

        if param_nonfinite:
            first_failure = ("Optimizer Step", f"param({param_nonfinite})", b_idx, glob_step)
            break

        # EMA update
        alpha = min(1.0 - 1.0 / (epoch + 1), theta)
        for p_tea, p_stu in zip(model_tea.parameters(), model_stu.parameters()):
            p_tea.data.mul_(alpha).add_(p_stu.data, alpha=1.0 - alpha)

        log_msg(f"  Batch {b_idx:03d}/{len(train_loader)}: TotalLoss={total_loss.item():.4f} (Seg={seg_loss.item():.4f}, Cons={consistency_loss.item():.4f}), MaxGradNorm={max_grad_norm:.3f}")

    log_msg("\n" + "=" * 80)
    if first_failure:
        log_msg(f"RESULT: First Non-Finite Stage = {first_failure[0]}, Target = {first_failure[1]}, Batch Index = {first_failure[2]}, Global Step = {first_failure[3]}")
    else:
        log_msg("RESULT: All 132 batches completed finite without error in this run!")
    log_msg("=" * 80)


if __name__ == "__main__":
    run_forensic_trace()
