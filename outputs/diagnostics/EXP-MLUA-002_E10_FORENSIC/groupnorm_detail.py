"""
EXP-MLUA-002: GROUPNORM INPUT & ACTIVATION INSPECTION ON BATCH 68
Inspects the exact input to GroupNorm in seg_blocks[0] on Image 3.
"""

import sys
import math
from pathlib import Path
import yaml
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
log_file = diag_dir / "groupnorm_detail_log.txt"

def log_msg(msg: str):
    print(msg, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def main():
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("GROUPNORM DETAIL TRACE\n")

    cfg_path = base_dir / "configs" / "experiments" / "EXP-MLUA-002.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

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

    ckpt_path = base_dir / "outputs" / "experiments" / "EXP-MLUA-002" / "checkpoints" / "EXP-MLUA-002_LATEST.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_stu.load_state_dict(ckpt["model_stu_state_dict"])
    model_tea.load_state_dict(ckpt["model_tea_state_dict"])

    optimizer = torch.optim.AdamW(model_stu.parameters(), lr=cfg["optimization"]["learning_rate"], weight_decay=cfg["optimization"]["weight_decay"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    theta = cfg["ssl"]["ema_theta"]
    mc_t = cfg["ssl"]["mc_iterations"]
    noise_sigma = cfg["ssl"]["noise_sigma"]
    noise_clamp = cfg["ssl"]["noise_clamp"]
    l_batch_size = cfg["data"]["labeled_batch_size"]
    epoch = 9

    model_stu.train()
    model_tea.eval()

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
                all_9 = all_fused_tea.view(l_batch_size, mc_t + 1, 1, 384, 384)
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

        # BATCH 68 INSPECTION
        log_msg("[BATCH 68 FOUND]")
        ul_imgs = imgs[l_batch_size:]
        repeated_ul = ul_imgs.repeat_interleave(mc_t + 1, dim=0)
        noise = torch.clamp(torch.randn_like(repeated_ul) * noise_sigma, -noise_clamp, noise_clamp)
        is_orig_mask = (torch.arange(repeated_ul.size(0), device=device) % (mc_t + 1) == 0).view(-1, 1, 1, 1)
        pert_ul_all = torch.where(is_orig_mask, repeated_ul, repeated_ul + noise)

        # Inspect Image 3 specifically
        img3_9perts = pert_ul_all[27:36] # [9, 1, 384, 384]
        log_msg(f"img3_9perts shape: {list(img3_9perts.shape)}, min={img3_9perts.min():.4f}, max={img3_9perts.max():.4f}")

        with torch.inference_mode():
            features = model_tea.encoder(img3_9perts)
            for f_i, feat in enumerate(features):
                log_msg(f"  Encoder c{f_i} shape={list(feat.shape)}, min={feat.min():.4f}, max={feat.max():.4f}, is_finite={torch.isfinite(feat).all().item()}")

            c2, c3, c4, c5 = features[-4:]
            p5 = model_tea.decoder.p5(c5)
            log_msg(f"  p5 shape={list(p5.shape)}, min={p5.min():.4f}, max={p5.max():.4f}, is_finite={torch.isfinite(p5).all().item()}")

            conv_op = model_tea.decoder.seg_blocks[0].block[0].block[0]
            gn_op = model_tea.decoder.seg_blocks[0].block[0].block[1]

            conv_out = conv_op(p5)
            log_msg(f"  conv_out shape={list(conv_out.shape)}, min={conv_out.min():.4f}, max={conv_out.max():.4f}, is_finite={torch.isfinite(conv_out).all().item()}")

            log_msg(f"  GroupNorm num_groups={gn_op.num_groups}, num_channels={gn_op.num_channels}, eps={gn_op.eps}")
            log_msg(f"  GroupNorm weight is_finite={torch.isfinite(gn_op.weight).all().item()}, min={gn_op.weight.min():.4f}, max={gn_op.weight.max():.4f}")
            log_msg(f"  GroupNorm bias is_finite={torch.isfinite(gn_op.bias).all().item()}, min={gn_op.bias.min():.4f}, max={gn_op.bias.max():.4f}")

            # Manual GroupNorm computation on conv_out
            N, C, H, W = conv_out.shape
            G = gn_op.num_groups
            x_reshaped = conv_out.view(N, G, C // G, H, W)
            mean = x_reshaped.mean(dim=[2, 3, 4], keepdim=True)
            var = x_reshaped.var(dim=[2, 3, 4], keepdim=True, unbiased=False)
            log_msg(f"  GroupNorm calculated mean min={mean.min():.4e}, max={mean.max():.4e}, is_finite={torch.isfinite(mean).all().item()}")
            log_msg(f"  GroupNorm calculated var min={var.min():.4e}, max={var.max():.4e}, is_finite={torch.isfinite(var).all().item()}")

            gn_out = gn_op(conv_out)
            log_msg(f"  gn_out is_finite={torch.isfinite(gn_out).all().item()}, NaNs={torch.isnan(gn_out).sum().item()}")

        break

if __name__ == "__main__":
    main()
