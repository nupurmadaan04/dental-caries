"""
EXP-MLUA-002: BATCH 68 DEEP-DIVE FORENSIC SCRIPT
Traces internal activations of model_tea on Batch 68 to pinpoint the exact layer producing NaN.
"""

import sys
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
from src.mlua.engine.train_exp002 import CachedTrainDataset, seed_everything

diag_dir = base_dir / "outputs" / "diagnostics" / "EXP-MLUA-002_E10_FORENSIC"
log_file = diag_dir / "batch68_deep_dive_log.txt"


def log_msg(msg: str):
    print(msg, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run_batch68_trace():
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("EXP-MLUA-002: BATCH 68 INTERNAL ACTIVATION DEEP-DIVE\n" + "=" * 80 + "\n")

    log_msg("=" * 80)
    log_msg("EXP-MLUA-002: BATCH 68 INTERNAL ACTIVATION DEEP-DIVE")
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

    ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
    model_stu.load_state_dict(ckpt["model_stu_state_dict"])
    model_tea.load_state_dict(ckpt["model_tea_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    theta = cfg["ssl"]["ema_theta"]
    mc_t = cfg["ssl"]["mc_iterations"]
    noise_sigma = cfg["ssl"]["noise_sigma"]
    noise_clamp = cfg["ssl"]["noise_clamp"]
    transize = cfg["data"]["patch_size"]
    epoch = 9

    model_stu.train()
    model_tea.eval()

    # Step through batches 0 to 67 to reach exact state at batch 68
    log_msg("[Simulation] Fast-forwarding training to Batch 68...")
    for b_idx, (imgs, gts) in enumerate(train_loader):
        glob_step = epoch * len(train_loader) + b_idx + 1
        imgs = imgs.to(device)
        gts = gts.to(device)

        if b_idx < 68:
            # Standard step
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

        # WE ARE AT BATCH 68!
        log_msg("\n" + "=" * 80)
        log_msg(f"[BATCH 68 DEEP DIVE] Global Step = {glob_step}")
        log_msg("=" * 80)

        log_msg(f"imgs: shape={list(imgs.shape)}, min={imgs.min().item():.4f}, max={imgs.max().item():.4f}, is_finite={torch.isfinite(imgs).all().item()}")
        log_msg(f"gts: shape={list(gts.shape)}, min={gts.min().item():.4f}, max={gts.max().item():.4f}, is_finite={torch.isfinite(gts).all().item()}")

        # Check teacher weights before forward pass
        tea_params_finite = True
        for name, param in model_tea.named_parameters():
            if not torch.isfinite(param).all():
                log_msg(f"  [TEA PARAM NON-FINITE BEFORE FORWARD] {name}: NaNs={torch.isnan(param).sum().item()}, Infs={torch.isinf(param).sum().item()}")
                tea_params_finite = False
        log_msg(f"Teacher all parameters finite before forward: {tea_params_finite}")

        # Check teacher buffers (BatchNorm running_mean, running_var)
        tea_buffers_finite = True
        for name, buf in model_tea.named_buffers():
            if not torch.isfinite(buf).all():
                log_msg(f"  [TEA BUFFER NON-FINITE] {name}: NaNs={torch.isnan(buf).sum().item()}, Infs={torch.isinf(buf).sum().item()}")
                tea_buffers_finite = False
        log_msg(f"Teacher all buffers finite: {tea_buffers_finite}")

        ul_imgs = imgs[l_batch_size:] # [4, 1, 384, 384]
        for u_i in range(4):
            log_msg(f"Unlabeled image {u_i}: min={ul_imgs[u_i].min().item():.4f}, max={ul_imgs[u_i].max().item():.4f}, mean={ul_imgs[u_i].mean().item():.4f}, is_finite={torch.isfinite(ul_imgs[u_i]).all().item()}")

        repeated_ul = ul_imgs.repeat_interleave(mc_t + 1, dim=0)
        noise = torch.clamp(torch.randn_like(repeated_ul) * noise_sigma, -noise_clamp, noise_clamp)
        is_orig_mask = (torch.arange(repeated_ul.size(0), device=device) % (mc_t + 1) == 0).view(-1, 1, 1, 1)
        pert_ul_all = torch.where(is_orig_mask, repeated_ul, repeated_ul + noise)
        log_msg(f"pert_ul_all: shape={list(pert_ul_all.shape)}, min={pert_ul_all.min().item():.4f}, max={pert_ul_all.max().item():.4f}, is_finite={torch.isfinite(pert_ul_all).all().item()}")

        # Test each unlabeled image individually through model_tea
        log_msg("\n[Testing Individual Unlabeled Images Through Teacher Model]")
        for u_i in range(4):
            single_pert = pert_ul_all[u_i * 9 : (u_i + 1) * 9] # [9, 1, 384, 384]
            with torch.inference_mode():
                out_fused, out_aux = model_tea(single_pert)
                nan_fused = torch.isnan(out_fused).sum().item()
                inf_fused = torch.isinf(out_fused).sum().item()
                log_msg(f"  Unlabeled image {u_i} (9 perts) -> fused output NaNs={nan_fused}, Infs={inf_fused}, min={out_fused.min().item() if nan_fused==0 else 'NaN'}, max={out_fused.max().item() if nan_fused==0 else 'NaN'}")

        # Register forward hooks on all layers of model_tea to isolate the EXACT module producing NaN
        log_msg("\n[Tracing Layer-by-Layer Activations of model_tea with Hooks]")
        hook_log = []

        def get_hook(mod_name):
            def hook_fn(module, inp, outp):
                if isinstance(outp, tuple):
                    for idx, o in enumerate(outp):
                        if isinstance(o, torch.Tensor):
                            nan_c = torch.isnan(o).sum().item()
                            inf_c = torch.isinf(o).sum().item()
                            finite = torch.isfinite(o).all().item()
                            hook_log.append((f"{mod_name}[{idx}]", list(o.shape), finite, nan_c, inf_c))
                elif isinstance(outp, torch.Tensor):
                    nan_c = torch.isnan(outp).sum().item()
                    inf_c = torch.isinf(outp).sum().item()
                    finite = torch.isfinite(outp).all().item()
                    hook_log.append((mod_name, list(outp.shape), finite, nan_c, inf_c))
            return hook_fn

        hooks = []
        for name, module in model_tea.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                hooks.append(module.register_forward_hook(get_hook(name)))

        with torch.inference_mode():
            all_fused_tea, _ = model_tea(pert_ul_all)

        for h in hooks:
            h.remove()

        first_nan_layer = None
        for name, shape, finite, nans, infs in hook_log:
            status = "FINITE" if finite else f"NON-FINITE (NaNs={nans}, Infs={infs})"
            if not finite and first_nan_layer is None:
                first_nan_layer = (name, shape, nans, infs)
                log_msg(f"  >>> FIRST NON-FINITE MODULE: {name} | shape={shape} | NaNs={nans}, Infs={infs}")
            else:
                if not finite:
                    log_msg(f"      Subsequent non-finite: {name} | shape={shape} | NaNs={nans}, Infs={infs}")

        log_msg("\n" + "=" * 80)
        log_msg(f"DEEP DIVE CONCLUSION: First non-finite module inside Teacher Model = {first_nan_layer}")
        log_msg("=" * 80)
        break


if __name__ == "__main__":
    run_batch68_trace()
