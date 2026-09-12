"""
EXP-MLUA-001 Official Training Execution Script
Implements:
1. Source-Verified Official MLUA 10% SSL Baseline (ResNet-34 FPN, DC1000 dataset, MAX_EPOCHS=200).
2. Day 2 Verified Compute Optimization: Unified 36-image Teacher Forward Pass under torch.inference_mode.
3. RAM Pre-cached Dataset (~704 MB) for zero disk I/O bottlenecks.
4. In-Place Vectorized EMA updates.
5. Resume-safe checkpointing (LATEST vs BEST) with full recovery and integrity validation.
6. Safe Time-budgeted Execution (9-hour window with clean epoch boundary stopping).
7. Full CSV history tracking and live TRAINING_STATUS.md generation.
"""

import os
import sys
import time
import json
import shutil
import hashlib
import platform
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List, Optional

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms as T

# Base project directory
base_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(base_dir))

from src.mlua.models.fpn import Net
from src.mlua.data.sampler import TwoStreamBatchSampler
from util.utils import DiceLoss, sigmoid_mse_loss, sigmoid_rampup, get_current_consistency_weight


def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class CachedTrainDataset(Dataset):
    """
    RAM-cached dataset holding raw decoded uint8 images and masks (~704 MB total).
    Dynamic spatial and photometric augmentations are generated per-sample on the fly,
    strictly preserving exact data randomness, distribution, and augmentation schedules.
    """
    def __init__(self, image_list: List[Path], label_list: List[Path], transize: int = 384):
        self.transize = transize
        self.data_paths = list(zip(image_list, label_list))
        self.cached_images: List[Image.Image] = []
        self.cached_labels: List[Image.Image] = []

        self.img_transform = T.Compose([
            T.ColorJitter(brightness=0.5, contrast=0.5),
        ])
        self.both_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(45),
        ])
        self.resize_transform = T.Resize((self.transize, self.transize))
        self.normalize_transform = T.ToTensor()

        print(f"[Dataset] Pre-caching {len(self.data_paths)} training patches into RAM...", flush=True)
        t0 = time.time()
        for img_path, lbl_path in self.data_paths:
            img = Image.open(str(img_path)).convert("L")
            if lbl_path is not None and lbl_path.exists():
                lbl = Image.open(str(lbl_path)).convert("L")
            else:
                lbl = Image.fromarray(np.zeros((self.transize, self.transize), dtype=np.uint8))
            self.cached_images.append(img)
            self.cached_labels.append(lbl)
        print(f"[Dataset] Pre-caching completed in {time.time() - t0:.2f}s (~704 MB RAM).", flush=True)

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.cached_images[index].copy()
        label = self.cached_labels[index].copy()

        import random
        seed = random.randint(0, 100000)
        torch.random.manual_seed(seed)
        image = self.both_transform(image)
        torch.random.manual_seed(seed)
        label = self.both_transform(label)

        image = self.img_transform(image)
        image = self.resize_transform(image)
        label = self.resize_transform(label)

        image_tensor = self.normalize_transform(image)
        label_tensor = self.normalize_transform(label)

        return image_tensor, label_tensor


class DeterministicSubsetDataset(Dataset):
    """
    Deterministic evaluation dataset pointing to pre-cached RAM raw images and masks.
    Applies ONLY deterministic resize (384x384) and ToTensor normalization.
    Strictly ZERO random augmentations (no flip, no rotation, no color jitter).
    """
    def __init__(self, parent_dataset: CachedTrainDataset, indices: List[int]):
        self.parent = parent_dataset
        self.indices = indices
        self.resize_transform = T.Resize((parent_dataset.transize, parent_dataset.transize))
        self.normalize_transform = T.ToTensor()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]
        image = self.parent.cached_images[real_idx].copy()
        label = self.parent.cached_labels[real_idx].copy()

        image = self.resize_transform(image)
        label = self.resize_transform(label)

        image_tensor = self.normalize_transform(image)
        label_tensor = self.normalize_transform(label)

        return image_tensor, label_tensor


def calculate_metrics_batch(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Calculates Dice, IoU, Precision, Recall/Sensitivity, Specificity, F1."""
    preds_bin = (preds > threshold).float().view(-1)
    targets_bin = (targets > threshold).float().view(-1)

    tp = (preds_bin * targets_bin).sum().item()
    fp = (preds_bin * (1.0 - targets_bin)).sum().item()
    fn = ((1.0 - preds_bin) * targets_bin).sum().item()
    tn = ((1.0 - preds_bin) * (1.0 - targets_bin)).sum().item()

    eps = 1e-4
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    spec = tn / (tn + fp + eps)
    f1 = dice # F1 is mathematically identical to Dice score
    pred_prev = preds_bin.mean().item()
    gt_prev = targets_bin.mean().item()

    return {
        "acc": float(acc),
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "f1": float(f1),
        "pred_prevalence": float(pred_prev),
        "gt_prevalence": float(gt_prev),
    }


def update_training_status_md(
    status_file: Path,
    exp_id: str,
    completed_epoch: int,
    max_epochs: int,
    best_epoch: int,
    best_dice: float,
    current_dice: float,
    total_runtime_sec: float,
    checkpoints_dir: Path,
    is_complete: bool,
    status_text: str,
):
    """Generates the live TRAINING_STATUS.md file."""
    runtime_str = str(timedelta(seconds=int(total_runtime_sec)))
    next_epoch = completed_epoch + 1 if completed_epoch < max_epochs else "N/A (Complete)"

    md_content = f"""# EXP-MLUA-001 Training Status

| Property | Value |
|---|---|
| **Experiment ID** | `{exp_id}` |
| **Architecture** | ResNet-34 FPN (4 Auxiliary Heads + 1 Fused Head) |
| **Dataset** | DC1000 Training Patches (2,389 total) |
| **Labeled / Unlabeled** | 265 Labeled (10%) / 2,124 Unlabeled (90%) |
| **Current Completed Epoch** | **Epoch {completed_epoch} / {max_epochs}** |
| **Maximum Intended Epochs** | {max_epochs} |
| **Best Epoch** | **Epoch {best_epoch}** |
| **Best Validation Dice** | **{best_dice:.4f}** ({best_dice*100:.2f}%) |
| **Current Validation Dice** | **{current_dice:.4f}** ({current_dice*100:.2f}%) |
| **Total Cumulative Runtime** | `{runtime_str}` |
| **Run Status** | **`{status_text}`** |
| **Resumable** | **`YES`** |
| **Next Resume Epoch** | **Epoch {next_epoch}** |
| **BEST Checkpoint** | `{checkpoints_dir / f"{exp_id}_BEST.pth"}` |
| **LATEST Checkpoint** | `{checkpoints_dir / f"{exp_id}_LATEST.pth"}` |

---
*Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    with open(status_file, "w") as f:
        f.write(md_content)


def main():
    start_session_time = time.time()
    time_budget_sec = 9.0 * 3600.0 # 9-hour safe execution budget
    safe_buffer_sec = 25.0 * 60.0  # 25-minute buffer to ensure clean epoch completion

    print("\n" + "=" * 75, flush=True)
    print("EXP-MLUA-001: OFFICIAL 10% SSL RESUME-SAFE TRAINING RUN", flush=True)
    print("=" * 75, flush=True)

    # 1. Configuration & Directories
    config_file = base_dir / "configs" / "mlua_default.yaml"
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    exp_dir = base_dir / "outputs" / "experiments" / "EXP-MLUA-001"
    checkpoints_dir = exp_dir / "checkpoints"
    root_checkpoints_dir = base_dir / "checkpoints"
    metrics_dir = exp_dir / "metrics"
    logs_dir = exp_dir / "logs"

    for d in [checkpoints_dir, root_checkpoints_dir, metrics_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    status_file = exp_dir / "TRAINING_STATUS.md"
    history_csv = exp_dir / "EXP-MLUA-001_FULL_TRAINING_HISTORY.csv"
    latest_ckpt_path = checkpoints_dir / "EXP-MLUA-001_LATEST.pth"
    best_ckpt_path = checkpoints_dir / "EXP-MLUA-001_BEST.pth"

    seed = cfg["experiment"]["seed"]
    seed_everything(seed)
    torch.set_num_threads(8)

    device = torch.device("cpu")
    print(f"[Device] Target Compute Device: {device} (8 OpenMP/MKL Worker Threads)", flush=True)

    # 2. Data Loading & Partitioning
    train_img_dir = base_dir / cfg["data"]["train_image_dir"]
    train_lbl_dir = base_dir / cfg["data"]["train_label_dir"]
    train_img_files = sorted(list(train_img_dir.glob("*.png")), key=lambda x: int(x.stem))
    train_lbl_files = sorted(list(train_lbl_dir.glob("*.png")), key=lambda x: int(x.stem))

    assert len(train_img_files) == 2389, f"Expected 2389 images, found {len(train_img_files)}"
    assert len(train_lbl_files) == 2389, f"Expected 2389 labels, found {len(train_lbl_files)}"

    labeled_count = cfg["data"]["labeled_rates"]["0.1"] # 265
    all_indices = list(range(len(train_img_files)))
    labeled_indices = all_indices[:labeled_count]
    unlabeled_indices = all_indices[labeled_count:]

    batch_size = cfg["data"]["batch_size"] # 8
    l_batch_size = cfg["data"]["labeled_batch_size"] # 4
    batch_sampler = TwoStreamBatchSampler(
        labeled_indices,
        unlabeled_indices,
        batch_size=batch_size,
        l_batch_size=l_batch_size,
    )

    train_dataset = CachedTrainDataset(train_img_files, train_lbl_files, transize=cfg["data"]["patch_size"])
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=0)

    val_indices = labeled_indices[:50]
    val_sub_dataset = DeterministicSubsetDataset(train_dataset, val_indices)
    val_loader = DataLoader(val_sub_dataset, batch_size=4, shuffle=False, num_workers=0)

    # 3. Model Architecture & Optimizers
    model_stu = Net(in_c=1, out_c=1, encoder_weights=None).to(device)
    model_tea = Net(in_c=1, out_c=1, encoder_weights=None).to(device)

    for p in model_tea.parameters():
        p.requires_grad = False

    lr = cfg["optimization"]["learning_rate"]
    weight_decay = cfg["optimization"]["weight_decay"]
    optimizer = torch.optim.AdamW(model_stu.parameters(), lr=lr, weight_decay=weight_decay)

    max_epochs = cfg["experiment"]["max_epochs"] # 200
    poly_lr_fn = lambda epoch: (1.0 - float(epoch) / max_epochs) ** cfg["optimization"]["poly_power"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr_fn)

    bce_loss_fn = F.binary_cross_entropy_with_logits
    dice_loss_fn = DiceLoss()

    theta = cfg["ssl"]["ema_theta"] # 0.99
    mc_t = cfg["ssl"]["mc_iterations"] # 8
    noise_sigma = cfg["ssl"]["noise_sigma"] # 0.01
    noise_clamp = cfg["ssl"]["noise_clamp"] # 0.1

    # 4. Checkpoint Resume Logic
    start_epoch = 0
    best_val_dice = -1.0
    best_epoch = 0
    best_metrics = {}
    history_records: List[Dict[str, Any]] = []
    cumulative_runtime_sec = 0.0

    if history_csv.exists():
        try:
            df_existing = pd.read_csv(history_csv)
            history_records = df_existing.to_dict("records")
            if len(history_records) > 0:
                cumulative_runtime_sec = float(df_existing["epoch_duration"].sum())
        except Exception as e:
            print(f"[Warning] Failed to read existing history CSV: {e}", flush=True)

    if latest_ckpt_path.exists():
        print(f"\n[Resume] Found existing LATEST checkpoint: {latest_ckpt_path}", flush=True)
        try:
            ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
            completed_epoch = ckpt["epoch"]
            model_stu.load_state_dict(ckpt["model_stu_state_dict"])
            model_tea.load_state_dict(ckpt["model_tea_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            
            start_epoch = completed_epoch
            print(f"[Resume] Successfully loaded state! Completed Epoch: {completed_epoch}. Resuming from Epoch {start_epoch + 1} -> {max_epochs}.", flush=True)
            
            # Check BEST checkpoint
            if best_ckpt_path.exists():
                ckpt_b = torch.load(best_ckpt_path, map_location=device, weights_only=False)
                best_epoch = ckpt_b.get("epoch", 1)
                best_metrics = ckpt_b.get("metrics", {})
                best_val_dice = best_metrics.get("val_dice", 0.0)
                print(f"[Resume] Best Checkpoint: Epoch {best_epoch} with Val Dice {best_val_dice:.4f}", flush=True)
        except Exception as e:
            print(f"[Resume Error] Failed loading existing checkpoint ({e}). Starting fresh.", flush=True)
            start_epoch = 0
    else:
        print("[Init] No existing checkpoint found. Initializing student and teacher from scratch.", flush=True)
        for p_tea, p_stu in zip(model_tea.parameters(), model_stu.parameters()):
            p_tea.data.copy_(p_stu.data)

    # 5. PRE-FLIGHT CHECK REPORT
    print("\n" + "=" * 75, flush=True)
    print("PRE-FLIGHT VERIFICATION REPORT", flush=True)
    print("=" * 75, flush=True)
    print(f"• Experiment ID         : EXP-MLUA-001")
    print(f"• Existing Checkpoint   : {'Found (Epoch ' + str(start_epoch) + ')' if start_epoch > 0 else 'None (Clean Start)'}")
    print(f"• Resume Epoch Target   : Epoch {start_epoch + 1}")
    print(f"• Intended Max Epochs   : {max_epochs}")
    print(f"• Training Dataset      : {len(train_dataset)} patches (265 Labeled [10%] / 2,124 Unlabeled [90%])")
    print(f"• Batch Composition     : {batch_size} total (4 Labeled + 4 Unlabeled)")
    print(f"• Compute Optimization  : Day-2 Unified 36-Image Teacher Forward Pass (torch.inference_mode)")
    print(f"• Expected Epoch Time   : ~18.39 minutes (66 batches)")
    print(f"• Safe Execution Budget : {time_budget_sec / 3600.0:.1f} hours (~25-30 epochs)")
    print(f"• Sealed Test Set       : SEALED & UNTOUCHED (dataset/test/ protected)")
    print("=" * 75 + "\n", flush=True)

    # 6. TRAINING EXECUTION LOOP
    glob_step = start_epoch * len(train_loader)
    stopped_due_to_budget = False

    for epoch in range(start_epoch, max_epochs):
        epoch_num = epoch + 1
        elapsed_session = time.time() - start_session_time

        # Time budget check before starting epoch
        estimated_epoch_sec = 1100.0 # ~18.3 min
        if (elapsed_session + estimated_epoch_sec) > (time_budget_sec - safe_buffer_sec) and epoch > start_epoch:
            print(f"\n[Time Budget] Approaching safe {time_budget_sec/3600.0:.1f}h execution window. Terminating cleanly after Epoch {epoch}.", flush=True)
            stopped_due_to_budget = True
            break

        epoch_start_time = time.time()
        model_stu.train()
        model_tea.eval()

        running_train_loss = 0.0
        running_seg_loss = 0.0
        running_cons_loss = 0.0
        num_batches = 0

        for batch_idx, (imgs, gts) in enumerate(train_loader):
            glob_step += 1
            imgs = imgs.unsqueeze(1) if imgs.ndim == 3 else imgs
            gts = gts.unsqueeze(1) if gts.ndim == 3 else gts
            imgs = imgs.to(device)
            gts = gts.to(device)

            # Student forward pass
            pred_fused, pred_aux_list = model_stu(imgs)

            # Unified 36-Image Teacher Forward Pass (Day-2 Verified Optimization)
            ul_data = imgs[l_batch_size:] # [4, 1, 384, 384]
            noise_base = torch.clamp(torch.randn_like(ul_data) * noise_sigma, -noise_clamp, noise_clamp)
            noises_mc = torch.clamp(
                torch.randn(mc_t, *ul_data.shape, device=device) * noise_sigma,
                -noise_clamp,
                noise_clamp
            )

            with torch.inference_mode():
                stride = ul_data.shape[0]
                unified_in = torch.empty((stride + mc_t * stride, 1, 384, 384), device=device)
                unified_in[:stride] = ul_data + noise_base
                unified_in[stride:] = (ul_data.unsqueeze(0) + noises_mc).view(mc_t * stride, 1, 384, 384)

                all_final, all_pyramid = model_tea(unified_in)
                ul_pred_tea = all_final[:stride]
                mc_final = all_final[stride:]
                mc_pyr = [h[stride:] for h in all_pyramid]

                b_final_5 = mc_final.view(mc_t, stride, 1, 384, 384).unsqueeze(0)
                b_pyr_5 = torch.stack([h.view(mc_t, stride, 1, 384, 384) for h in mc_pyr], dim=0)
                all_5 = torch.cat([b_final_5, b_pyr_5], dim=0).view(5 * mc_t, stride, 1, 384, 384)
                mean_preds = torch.mean(all_5, dim=0).sigmoid()
                uncertainty = -2.0 * torch.sum(mean_preds * torch.log(mean_preds + 1e-6), dim=1, keepdim=True)

            threshold = (cfg["ssl"]["threshold_start_factor"] + (cfg["ssl"]["threshold_end_factor"] - cfg["ssl"]["threshold_start_factor"]) * sigmoid_rampup(glob_step, cfg["ssl"]["threshold_rampup_steps"])) * np.log(2)
            mask = (uncertainty < threshold).float()

            consistency_dist = sigmoid_mse_loss(pred_fused[l_batch_size:], ul_pred_tea)
            consistency_loss = torch.sum(mask * consistency_dist) / (2.0 * torch.sum(mask) + 1e-16)

            # Deep supervision loss
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

            # Backward pass & step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Vectorized in-place EMA
            alpha = min(1.0 - 1.0 / (epoch + 1), theta)
            for p_tea, p_stu in zip(model_tea.parameters(), model_stu.parameters()):
                p_tea.data.mul_(alpha).add_(p_stu.data, alpha=1.0 - alpha)

            running_train_loss += total_loss.item()
            running_seg_loss += seg_loss.item()
            running_cons_loss += consistency_loss.item()
            num_batches += 1

        scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_duration = time.time() - epoch_start_time
        cumulative_runtime_sec += epoch_duration

        epoch_train_loss = running_train_loss / max(num_batches, 1)
        epoch_seg_loss = running_seg_loss / max(num_batches, 1)
        epoch_cons_loss = running_cons_loss / max(num_batches, 1)

        # 7. Validation Phase
        model_stu.eval()
        val_losses = []
        val_dices = []
        val_ious = []
        val_precs = []
        val_recs = []
        val_specs = []
        val_f1s = []
        val_max_probs = []
        val_pred_prevs = []

        with torch.no_grad():
            for v_imgs, v_gts in val_loader:
                v_imgs = v_imgs.unsqueeze(1) if v_imgs.ndim == 3 else v_imgs
                v_gts = v_gts.unsqueeze(1) if v_gts.ndim == 3 else v_gts
                v_imgs = v_imgs.to(device)
                v_gts = v_gts.to(device)

                v_pred_fused, _ = model_stu(v_imgs)
                v_loss = bce_loss_fn(v_pred_fused, v_gts) + dice_loss_fn(v_pred_fused, v_gts)
                val_losses.append(v_loss.item())

                v_pred_sig = torch.sigmoid(v_pred_fused)
                metrics = calculate_metrics_batch(v_pred_sig, v_gts, threshold=cfg["evaluation"]["decision_threshold"])
                val_dices.append(metrics["dice"])
                val_ious.append(metrics["iou"])
                val_precs.append(metrics["precision"])
                val_recs.append(metrics["recall"])
                val_specs.append(metrics["specificity"])
                val_f1s.append(metrics["f1"])
                val_max_probs.append(v_pred_sig.max().item())
                val_pred_prevs.append(metrics["pred_prevalence"])

        mean_val_loss = float(np.mean(val_losses))
        mean_val_dice = float(np.mean(val_dices))
        mean_val_iou = float(np.mean(val_ious))
        mean_val_prec = float(np.mean(val_precs))
        mean_val_rec = float(np.mean(val_recs))
        mean_val_spec = float(np.mean(val_specs))
        mean_val_f1 = float(np.mean(val_f1s))
        max_foreground_prob = float(np.max(val_max_probs))
        mean_pred_prev = float(np.mean(val_pred_prevs))

        # Checkpoint NaN/Inf Validation
        has_nan = np.isnan(epoch_train_loss) or np.isnan(mean_val_loss) or np.isnan(mean_val_dice)
        if has_nan:
            print(f"[ERROR] NaN/Inf encountered at Epoch {epoch_num}! Halting to prevent corrupted weights.", flush=True)
            break

        epoch_record = {
            "epoch": epoch_num,
            "train_loss": round(epoch_train_loss, 5),
            "supervised_loss": round(epoch_seg_loss, 5),
            "consistency_loss": round(epoch_cons_loss, 5),
            "val_loss": round(mean_val_loss, 5),
            "val_dice": round(mean_val_dice, 5),
            "val_iou": round(mean_val_iou, 5),
            "val_precision": round(mean_val_prec, 5),
            "val_recall": round(mean_val_rec, 5),
            "val_specificity": round(mean_val_spec, 5),
            "val_f1": round(mean_val_f1, 5),
            "max_foreground_prob": round(max_foreground_prob, 5),
            "pred_prevalence": round(mean_pred_prev, 6),
            "learning_rate": round(current_lr, 7),
            "global_step": glob_step,
            "epoch_duration": round(epoch_duration, 2),
            "cumulative_runtime": round(cumulative_runtime_sec, 2),
        }

        # Update History CSV (Avoid duplicates)
        history_records = [r for r in history_records if r["epoch"] != epoch_num]
        history_records.append(epoch_record)
        df_hist = pd.DataFrame(history_records)
        df_hist.sort_values("epoch", inplace=True)
        df_hist.to_csv(history_csv, index=False)

        # 8. Checkpoint Saving (LATEST vs BEST)
        torch.save(
            {
                "epoch": epoch_num,
                "global_step": glob_step,
                "model_stu_state_dict": model_stu.state_dict(),
                "model_tea_state_dict": model_tea.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": cfg,
                "metrics": epoch_record,
                "best_epoch": best_epoch,
                "best_val_dice": best_val_dice,
            },
            latest_ckpt_path,
        )
        shutil.copyfile(latest_ckpt_path, root_checkpoints_dir / "EXP-MLUA-001_LATEST.pth")

        is_new_best = False
        if mean_val_dice > best_val_dice:
            best_val_dice = mean_val_dice
            best_epoch = epoch_num
            best_metrics = epoch_record.copy()
            torch.save(
                {
                    "epoch": best_epoch,
                    "global_step": glob_step,
                    "model_stu_state_dict": model_stu.state_dict(),
                    "model_tea_state_dict": model_tea.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                    "metrics": best_metrics,
                },
                best_ckpt_path,
            )
            shutil.copyfile(best_ckpt_path, root_checkpoints_dir / "EXP-MLUA-001_BEST.pth")
            is_new_best = True

        # Progress Output
        time_elapsed_str = str(timedelta(seconds=int(time.time() - start_session_time)))
        est_remaining_str = str(timedelta(seconds=int((max_epochs - epoch_num) * epoch_duration)))
        print(
            f"Epoch [{epoch_num:03d}/{max_epochs:03d}] | "
            f"Train: {epoch_train_loss:.4f} (Seg: {epoch_seg_loss:.4f}, Cons: {epoch_cons_loss:.4f}) | "
            f"Val Loss: {mean_val_loss:.4f} | Val Dice: {mean_val_dice:.4f} | Val IoU: {mean_val_iou:.4f} | "
            f"Val Prec: {mean_val_prec:.4f} | Val Rec: {mean_val_rec:.4f} | Val Spec: {mean_val_spec:.4f} | "
            f"Max Prob: {max_foreground_prob:.4f} | "
            f"LR: {current_lr:.6f} | Time: {epoch_duration:.1f}s ({epoch_duration/60:.2f}m) | "
            f"Elapsed: {time_elapsed_str} | Remaining: {est_remaining_str}"
            f"{' --> [NEW BEST]' if is_new_best else ''}",
            flush=True
        )

        # Update Live Markdown Status
        status_text = "COMPLETE" if epoch_num >= max_epochs else "RUNNING"
        update_training_status_md(
            status_file=status_file,
            exp_id="EXP-MLUA-001",
            completed_epoch=epoch_num,
            max_epochs=max_epochs,
            best_epoch=best_epoch,
            best_dice=best_val_dice,
            current_dice=mean_val_dice,
            total_runtime_sec=cumulative_runtime_sec,
            checkpoints_dir=checkpoints_dir,
            is_complete=(epoch_num >= max_epochs),
            status_text=status_text,
        )

    # 9. Final Training Wrap-up
    last_completed = history_records[-1]["epoch"] if len(history_records) > 0 else 0
    final_status = "COMPLETE" if last_completed >= max_epochs else "PARTIAL — RESUMABLE"
    
    update_training_status_md(
        status_file=status_file,
        exp_id="EXP-MLUA-001",
        completed_epoch=last_completed,
        max_epochs=max_epochs,
        best_epoch=best_epoch,
        best_dice=best_val_dice,
        current_dice=history_records[-1]["val_dice"] if len(history_records) > 0 else 0.0,
        total_runtime_sec=cumulative_runtime_sec,
        checkpoints_dir=checkpoints_dir,
        is_complete=(last_completed >= max_epochs),
        status_text=final_status,
    )

    print("\n" + "=" * 75, flush=True)
    print(f"TRAINING EXECUTION PAUSED / COMPLETED ({final_status})", flush=True)
    print(f"• Completed Epochs     : {last_completed} / {max_epochs}")
    print(f"• Best Epoch           : Epoch {best_epoch} (Val Dice: {best_val_dice:.4f})")
    print(f"• Total Cumulative Time: {str(timedelta(seconds=int(cumulative_runtime_sec)))}")
    print(f"• Checkpoint Status    : LATEST saved at Epoch {last_completed}")
    print(f"• Next Resume Epoch    : Epoch {last_completed + 1}")
    print("=" * 75 + "\n", flush=True)


if __name__ == "__main__":
    main()
