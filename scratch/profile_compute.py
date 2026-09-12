import os
import sys
import time
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from src.mlua.models.fpn import Net
from src.mlua.data.dataset import TrainDataset
from src.mlua.data.sampler import TwoStreamBatchSampler
from util.utils import DiceLoss, sigmoid_mse_loss, sigmoid_rampup, get_current_consistency_weight

def profile():
    results = {}
    print("EXP-MLUA-001 DETAILED COMPUTE FORENSIC PROFILING", flush=True)

    device = torch.device("cpu")
    torch.set_num_threads(8)
    
    results["pytorch_version"] = torch.__version__
    results["intra_op_threads"] = torch.get_num_threads()
    results["inter_op_threads"] = torch.get_num_interop_threads()

    # 1. Dataset & Loading Profiling
    train_img_dir = base_dir / "data/raw/DC1000_dataset/train/images"
    train_lbl_dir = base_dir / "data/raw/DC1000_dataset/train/labels"
    train_img_files = sorted(list(train_img_dir.glob("*.png")), key=lambda x: int(x.stem))
    train_lbl_files = sorted(list(train_lbl_dir.glob("*.png")), key=lambda x: int(x.stem))

    dataset = TrainDataset(train_img_files, train_lbl_files, transize=384)
    labeled_indices = list(range(265))
    unlabeled_indices = list(range(265, 2389))

    batch_sampler = TwoStreamBatchSampler(labeled_indices, unlabeled_indices, batch_size=8, l_batch_size=4)
    train_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0)

    # Profile Data Loading & Augmentation (3 batches)
    t0 = time.time()
    for idx, (imgs, gts) in enumerate(train_loader):
        if idx >= 3:
            break
    avg_load_per_batch = (time.time() - t0) / 3
    results["avg_dataloader_batch_sec"] = avg_load_per_batch

    # Profile individual image decode & augmentation (5 items)
    t0 = time.time()
    for i in range(5):
        _ = dataset[i]
    t_single = (time.time() - t0) / 5
    results["single_image_getitem_sec"] = t_single

    # 2. Model Forward Profiling
    model_stu = Net(in_c=1, out_c=1).to(device)
    model_tea = Net(in_c=1, out_c=1).to(device)

    imgs = torch.randn(8, 1, 384, 384, device=device)
    gts = torch.zeros(8, 1, 384, 384, device=device)
    gts[:, :, 100:150, 100:150] = 1.0

    bce_loss_fn = F.binary_cross_entropy_with_logits
    dice_loss_fn = DiceLoss()
    optimizer = torch.optim.AdamW(model_stu.parameters(), lr=0.001, weight_decay=0.01)

    # Student Forward (2 runs)
    t0 = time.time()
    for _ in range(2):
        pred_fused, pred_aux_list = model_stu(imgs)
    t_stu_fwd = (time.time() - t0) / 2
    results["student_forward_8imgs_sec"] = t_stu_fwd

    # Student Backward (2 runs)
    t0 = time.time()
    for _ in range(2):
        pred_fused, pred_aux_list = model_stu(imgs)
        bce_loss = 0.0
        dice_loss = 0.0
        for pred_aux in pred_aux_list:
            bce_loss += bce_loss_fn(pred_aux[:4], gts[:4])
            dice_loss += dice_loss_fn(pred_aux[:4], gts[:4])
        bce_loss += bce_loss_fn(pred_fused[:4], gts[:4])
        dice_loss += dice_loss_fn(pred_fused[:4], gts[:4])
        seg_loss = 0.5 * (bce_loss / 4.0 + dice_loss / 4.0)
        optimizer.zero_grad()
        seg_loss.backward()
    t_stu_bwd = (time.time() - t0) / 2 - t_stu_fwd
    results["student_backward_sec"] = t_stu_bwd

    # 3. Teacher Forward & MC Passes
    ul_data = imgs[4:] # 4 images
    volume_batch_r = ul_data + torch.randn_like(ul_data) * 0.01

    # Baseline Teacher Forward (2 runs)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(2):
            ul_pred_tea = model_tea(volume_batch_r)[0]
    t_tea_baseline = (time.time() - t0) / 2
    results["teacher_baseline_4imgs_sec"] = t_tea_baseline

    # Sequential MC passes (T=8) (1 run)
    t0 = time.time()
    with torch.no_grad():
        for i in range(8):
            noise_i = torch.clamp(torch.randn_like(volume_batch_r) * 0.01, -0.1, 0.1)
            ema_inputs = ul_data + noise_i
            final_pred, pyramid_pred_list = model_tea(ema_inputs)
    t_mc_seq = time.time() - t0
    results["teacher_mc_sequential_T8_sec"] = t_mc_seq

    # Check Gradient Tracking during Teacher / MC
    with torch.no_grad():
        final_pred, pyr = model_tea(volume_batch_r)
        results["teacher_output_requires_grad"] = final_pred.requires_grad
        results["torch_is_grad_enabled_in_no_grad"] = torch.is_grad_enabled()

    # 4. Checkpoint Save Timing
    ckpt_path = base_dir / "scratch/profile_test.pth"
    t0 = time.time()
    torch.save({
        "epoch": 1,
        "model_stu": model_stu.state_dict(),
        "model_tea": model_tea.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, ckpt_path)
    t_ckpt = time.time() - t0
    ckpt_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    ckpt_path.unlink()
    results["checkpoint_save_sec"] = t_ckpt
    results["checkpoint_size_mb"] = ckpt_size_mb

    out_file = base_dir / "scratch/profile_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}", flush=True)

if __name__ == "__main__":
    profile()
