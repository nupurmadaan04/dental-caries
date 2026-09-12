"""
Read-Only Validation Cleanliness Sanity Check Script.
Evaluates Epoch 11 Checkpoint under:
1. OLD Stochastic Validation (with dynamic RandomRotation, RandomHorizontalFlip, ColorJitter)
2. NEW Deterministic Validation (zero stochastic augmentations, raw deterministic resize & normalize)
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

base_dir = Path("c:/Users/devin/MLUA")
sys.path.insert(0, str(base_dir))

from src.mlua.models.fpn import Net
from src.mlua.engine.train_exp001 import CachedTrainDataset, DeterministicSubsetDataset, calculate_metrics_batch
from util.utils import DiceLoss

def run_sanity_check():
    device = torch.device("cpu")
    ckpt_path = base_dir / "outputs" / "experiments" / "EXP-MLUA-001" / "checkpoints" / "EXP-MLUA-001_LATEST.pth"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    epoch = ckpt.get("epoch", "unknown")
    print(f"Loaded Checkpoint Epoch: {epoch}")

    model = Net(in_c=1, out_c=1, encoder_weights=None).to(device)
    model.load_state_dict(ckpt["model_stu_state_dict"])
    model.eval()

    train_img_dir = base_dir / "data" / "raw" / "DC1000_dataset" / "train" / "images"
    train_lbl_dir = base_dir / "data" / "raw" / "DC1000_dataset" / "train" / "labels"
    img_files = sorted(list(train_img_dir.glob("*.png")), key=lambda x: int(x.stem))
    lbl_files = sorted(list(train_lbl_dir.glob("*.png")), key=lambda x: int(x.stem))

    train_dataset = CachedTrainDataset(img_files, lbl_files, transize=384)
    labeled_indices = list(range(265))
    val_indices = labeled_indices[:50]

    bce_loss_fn = F.binary_cross_entropy_with_logits
    dice_loss_fn = DiceLoss()

    def evaluate_loader(loader, name: str):
        val_losses = []
        val_dices = []
        val_ious = []
        val_precs = []
        val_recs = []
        val_specs = []
        val_f1s = []
        max_probs = []
        mean_probs = []

        with torch.no_grad():
            for v_imgs, v_gts in loader:
                v_imgs = v_imgs.unsqueeze(1) if v_imgs.ndim == 3 else v_imgs
                v_gts = v_gts.unsqueeze(1) if v_gts.ndim == 3 else v_gts
                v_imgs = v_imgs.to(device)
                v_gts = v_gts.to(device)

                v_pred_fused, _ = model(v_imgs)
                v_loss = bce_loss_fn(v_pred_fused, v_gts) + dice_loss_fn(v_pred_fused, v_gts)
                val_losses.append(v_loss.item())

                v_pred_sig = torch.sigmoid(v_pred_fused)
                metrics = calculate_metrics_batch(v_pred_sig, v_gts, threshold=0.5)
                val_dices.append(metrics["dice"])
                val_ious.append(metrics["iou"])
                val_precs.append(metrics["precision"])
                val_recs.append(metrics["recall"])
                val_specs.append(metrics["specificity"])
                val_f1s.append(metrics["f1"])
                max_probs.append(v_pred_sig.max().item())
                mean_probs.append(v_pred_sig.mean().item())

        print(f"\n--- {name} Results (50 Validation Patches) ---")
        print(f"Val Loss        : {np.mean(val_losses):.5f}")
        print(f"Val Dice        : {np.mean(val_dices):.5f}")
        print(f"Val IoU         : {np.mean(val_ious):.5f}")
        print(f"Val Precision   : {np.mean(val_precs):.5f}")
        print(f"Val Recall      : {np.mean(val_recs):.5f}")
        print(f"Val Specificity : {np.mean(val_specs):.5f}")
        print(f"Val F1          : {np.mean(val_f1s):.5f}")
        print(f"Max Probability : {np.max(max_probs):.5f}")
        print(f"Mean Probability: {np.mean(mean_probs):.5f}")

        return {
            "val_loss": np.mean(val_losses),
            "val_dice": np.mean(val_dices),
            "val_iou": np.mean(val_ious),
            "val_precision": np.mean(val_precs),
            "val_recall": np.mean(val_recs),
            "val_specificity": np.mean(val_specs),
            "val_f1": np.mean(val_f1s),
            "max_prob": np.max(max_probs),
            "mean_prob": np.mean(mean_probs),
        }

    # 1. OLD Stochastic Validation
    old_sub_dataset = torch.utils.data.Subset(train_dataset, val_indices)
    old_loader = DataLoader(old_sub_dataset, batch_size=4, shuffle=False, num_workers=0)
    res_old = evaluate_loader(old_loader, "1. OLD Stochastic Validation (with RandomRotation/Flip/Jitter)")

    # 2. NEW Deterministic Validation
    new_sub_dataset = DeterministicSubsetDataset(train_dataset, val_indices)
    new_loader = DataLoader(new_sub_dataset, batch_size=4, shuffle=False, num_workers=0)
    res_new = evaluate_loader(new_loader, "2. NEW Deterministic Validation (Unaugmented Raw Patches)")

    return {"old": res_old, "new": res_new}

if __name__ == "__main__":
    run_sanity_check()
