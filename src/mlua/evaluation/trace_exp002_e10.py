"""
EXP-MLUA-002 Epoch 10 Diagnostic Forensic Script
Steps through batches of Epoch 10 to isolate the exact batch and loss component producing NaN.
"""

import os
import sys
import yaml
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

base_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(base_dir))

from src.mlua.models.fpn import Net
from src.mlua.data.sampler import TwoStreamBatchSampler
from util.utils import DiceLoss, sigmoid_mse_loss, sigmoid_rampup, get_current_consistency_weight
from src.mlua.engine.train_exp002 import CachedTrainDataset, DeterministicSubsetDataset, calculate_metrics_batch

def main():
    print("=" * 80)
    print("EXP-MLUA-002: EPOCH 10 FORENSIC BATCH TRACE")
    print("=" * 80)

    device = torch.device("cpu")
    torch.set_num_threads(8)

    cfg_path = base_dir / "configs" / "experiments" / "EXP-MLUA-002.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ckpt_path = base_dir / "outputs" / "experiments" / "EXP-MLUA-002" / "checkpoints" / "EXP-MLUA-002_LATEST.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"[Loaded Checkpoint] Epoch: {ckpt['epoch']}, Global Step: {ckpt['global_step']}")

    model_stu = Net(in_c=1, out_c=1, encoder_weights=None).to(device)
    model_tea = Net(in_c=1, out_c=1, encoder_weights=None).to(device)
    model_stu.load_state_dict(ckpt["model_stu_state_dict"])
    model_tea.load_state_dict(ckpt["model_tea_state_dict"])

    optimizer = torch.optim.AdamW(model_stu.parameters(), lr=cfg["optimization"]["learning_rate"], weight_decay=cfg["optimization"]["weight_decay"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    train_img_dir = base_dir / cfg["data"]["train_image_dir"]
    train_lbl_dir = base_dir / cfg["data"]["train_label_dir"]
    train_img_files = sorted(list(train_img_dir.glob("*.png")), key=lambda x: int(x.stem))
    train_lbl_files = sorted(list(train_lbl_dir.glob("*.png")), key=lambda x: int(x.stem))

    labeled_count = cfg["data"]["labeled_count"] # 530
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

    bce_loss_fn = F.binary_cross_entropy_with_logits
    dice_loss_fn = DiceLoss()

    theta = cfg["ssl"]["ema_theta"]
    mc_t = cfg["ssl"]["mc_iterations"]
    noise_sigma = cfg["ssl"]["noise_sigma"]
    noise_clamp = cfg["ssl"]["noise_clamp"]
    transize = cfg["data"]["patch_size"]
    l_batch_size = cfg["data"]["labeled_batch_size"]
    glob_step = ckpt["global_step"]
    epoch = 9 # 0-indexed for Epoch 10

    model_stu.train()
    model_tea.eval()

    print(f"\n[Starting Simulation of Epoch 10 Batches (Total: {len(train_loader)})...]")
    for b_idx, (imgs, gts) in enumerate(train_loader):
        glob_step += 1
        imgs = imgs.to(device)
        gts = gts.to(device)

        pred_fused, pred_aux_list = model_stu(imgs)

        # Teacher Pass
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
            bce_loss += bce_loss_fn(pred_aux[:l_batch_size], gts[:l_batch_size])
            dice_loss += dice_loss_fn(pred_aux[:l_batch_size], gts[:l_batch_size])
        bce_loss += bce_loss_fn(pred_fused[:l_batch_size], gts[:l_batch_size])
        dice_loss += dice_loss_fn(pred_fused[:l_batch_size], gts[:l_batch_size])
        seg_loss = 0.5 * (bce_loss / 4.0 + dice_loss / 4.0)

        consistency_weight = get_current_consistency_weight(epoch, cfg["ssl"]["consistency_rampup_epochs"], weight=cfg["ssl"]["consistency_weight_max"])
        total_loss = seg_loss + consistency_weight * consistency_loss

        t_val = total_loss.item()
        s_val = seg_loss.item()
        c_val = consistency_loss.item()

        if math.isnan(t_val) or math.isnan(s_val) or math.isnan(c_val) or math.isinf(t_val):
            print(f"[ALERT] Non-finite loss at Batch {b_idx:03d}!")
            print(f"  Total Loss: {t_val}, Seg Loss: {s_val}, Cons Loss: {c_val}")
            print(f"  BCE Loss: {bce_loss.item()}, Dice Loss: {dice_loss.item()}")
            print(f"  Mask Sum: {mask_sum.item()}, Threshold: {threshold}")
            print(f"  Pred Fused Min/Max: {pred_fused.min().item():.3f} / {pred_fused.max().item():.3f}")
            break

        # Backward pass & step
        optimizer.zero_grad()
        total_loss.backward()
        
        # Check grad norm
        total_grad_norm = 0.0
        has_nan_grad = False
        for p in model_stu.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                if math.isnan(param_norm) or math.isinf(param_norm):
                    has_nan_grad = True
                total_grad_norm += param_norm ** 2
        total_grad_norm = total_grad_norm ** 0.5

        if has_nan_grad:
            print(f"[ALERT] Non-finite gradient at Batch {b_idx:03d}! Grad Norm: {total_grad_norm}")
            break

        optimizer.step()

        # Vectorized in-place EMA
        alpha = min(1.0 - 1.0 / (epoch + 1), theta)
        for p_tea, p_stu in zip(model_tea.parameters(), model_stu.parameters()):
            p_tea.data.mul_(alpha).add_(p_stu.data, alpha=1.0 - alpha)

        if b_idx % 20 == 0 or b_idx == len(train_loader) - 1:
            print(f"  Batch {b_idx:03d}/{len(train_loader)}: Total Loss={t_val:.4f} (Seg={s_val:.4f}, Cons={c_val:.4f}), GradNorm={total_grad_norm:.3f}")

    print("\n[Epoch 10 Simulation Complete]")

if __name__ == "__main__":
    main()
