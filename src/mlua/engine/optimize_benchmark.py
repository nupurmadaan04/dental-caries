"""
DAY 1: EXP-MLUA-001 Compute Optimization & Numerical Equivalence Benchmark Script
Implements and benchmarks:
1. Sequential MC vs Batched MC Numerical Equivalence
2. Dataset RAM Caching Profiling
3. In-Place Vectorized EMA Update
4. Full Timing Decomposition on Fixed Deterministic Batch Subset
"""

import os
import sys
import time
import json
import psutil
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms as T

# Project base directory
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
        torch.cuda.manual_seed_all(seed)


class CachedTrainDataset(Dataset):
    """
    TrainDataset with RAM-cached decoded raw uint8 images and labels.
    Decodes all 2,389 PNG files once into RAM (~700 MB).
    Dynamic spatial and photometric augmentations are applied on-the-fly per access,
    strictly preserving exact data distribution and randomness.
    """
    def __init__(
        self,
        image_list: List[Path],
        label_list: List[Path],
        transize: int = 384,
        enable_cache: bool = True
    ):
        self.transize = transize
        self.data_paths = list(zip(image_list, label_list))
        self.enable_cache = enable_cache
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
        self.nomalize_transform = T.ToTensor()

        if self.enable_cache:
            print(f"Pre-caching {len(self.data_paths)} training patches into RAM...")
            t0 = time.time()
            for img_path, lbl_path in self.data_paths:
                img = Image.open(str(img_path)).convert("L")
                if lbl_path is not None and lbl_path.exists():
                    lbl = Image.open(str(lbl_path)).convert("L")
                else:
                    lbl = Image.fromarray(np.zeros((self.transize, self.transize), dtype=np.uint8))
                self.cached_images.append(img)
                self.cached_labels.append(lbl)
            print(f"Pre-caching completed in {time.time() - t0:.2f}s.")

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.enable_cache:
            image = self.cached_images[index].copy()
            label = self.cached_labels[index].copy()
        else:
            img_path, lbl_path = self.data_paths[index]
            image = Image.open(str(img_path)).convert("L")
            if lbl_path is not None and lbl_path.exists():
                label = Image.open(str(lbl_path)).convert("L")
            else:
                label = Image.fromarray(np.zeros((self.transize, self.transize), dtype=np.uint8))

        import random
        seed = random.randint(0, 10000)
        torch.random.manual_seed(seed)
        image = self.both_transform(image)
        torch.random.manual_seed(seed)
        label = self.both_transform(label)

        image = self.img_transform(image)
        image = self.resize_transform(image)
        label = self.resize_transform(label)

        image_tensor = self.nomalize_transform(image)
        label_tensor = self.nomalize_transform(label)

        return image_tensor, label_tensor


def run_numerical_equivalence_test():
    """
    Tests mathematical and numerical equivalence between:
    1. Sequential Teacher MC loop (T=8 forward passes of 4 images)
    2. Batched Teacher MC tensor pass (1 forward pass of 32 images)
    """
    print("\n" + "=" * 70)
    print("RUNNING OPTIMIZATION 1: NUMERICAL EQUIVALENCE TEST (SEQUENTIAL VS BATCHED MC)")
    print("=" * 70)

    seed_everything(42)
    device = torch.device("cpu")
    torch.set_num_threads(8)

    model_tea = Net(in_c=1, out_c=1, encoder_weights=None).to(device)
    model_tea.eval()
    for p in model_tea.parameters():
        p.requires_grad = False

    T_mc = 8
    B_ul = 4
    C, H, W = 1, 384, 384

    # Generate fixed deterministic input
    ul_data = torch.randn(B_ul, C, H, W, device=device)
    noise_sigma = 0.01
    noise_clamp = 0.1

    # Fixed perturbations for both methods
    torch.manual_seed(1234)
    noises = torch.clamp(torch.randn(T_mc, B_ul, C, H, W, device=device) * noise_sigma, -noise_clamp, noise_clamp)

    # -------------------------------------------------------------
    # 1. SEQUENTIAL MC IMPLEMENTATION (OLD)
    # -------------------------------------------------------------
    seq_preds = torch.zeros([T_mc * 5 * B_ul, C, H, W], device=device)
    with torch.no_grad():
        for i in range(T_mc):
            ema_inputs = ul_data + noises[i]
            final_pred, pyramid_pred_list = model_tea(ema_inputs)
            seq_preds[20 * i : 20 * i + 4] = final_pred
            pyramid_pred = torch.cat(pyramid_pred_list, dim=0)
            seq_preds[20 * i + 4 : 20 * i + 20] = pyramid_pred

        seq_preds = seq_preds.reshape(5 * T_mc, B_ul, C, H, W)
        seq_mean_preds = torch.mean(seq_preds, dim=0).sigmoid()
        seq_uncertainty = -2.0 * torch.sum(seq_mean_preds * torch.log(seq_mean_preds + 1e-6), dim=1, keepdim=True)

    # -------------------------------------------------------------
    # 2. BATCHED MC IMPLEMENTATION (NEW)
    # -------------------------------------------------------------
    with torch.no_grad():
        batched_inputs = (ul_data.unsqueeze(0) + noises).view(T_mc * B_ul, C, H, W)
        batched_final_pred, batched_pyramid_list = model_tea(batched_inputs)
        
        # Reshape final_pred: [32, 1, 384, 384] -> [1, T, B_ul, 1, 384, 384]
        b_final = batched_final_pred.view(T_mc, B_ul, C, H, W).unsqueeze(0)
        
        # Reshape pyramid_list: 4 heads each [32, 1, 384, 384] -> [4, T, B_ul, 1, 384, 384]
        b_pyramid = torch.stack([head.view(T_mc, B_ul, C, H, W) for head in batched_pyramid_list], dim=0)
        
        # Concatenate 5 heads -> [5, T, B_ul, 1, 384, 384] -> reshape to [5*T, B_ul, 1, 384, 384]
        b_all = torch.cat([b_final, b_pyramid], dim=0).view(5 * T_mc, B_ul, C, H, W)
        
        batched_mean_preds = torch.mean(b_all, dim=0).sigmoid()
        batched_uncertainty = -2.0 * torch.sum(batched_mean_preds * torch.log(batched_mean_preds + 1e-6), dim=1, keepdim=True)

    # -------------------------------------------------------------
    # 3. NUMERICAL EQUIVALENCE CHECKS
    # -------------------------------------------------------------
    diff_mean_preds = (seq_mean_preds - batched_mean_preds).abs()
    diff_uncertainty = (seq_uncertainty - batched_uncertainty).abs()

    max_diff_prob = diff_mean_preds.max().item()
    mean_diff_prob = diff_mean_preds.mean().item()
    max_diff_unc = diff_uncertainty.max().item()
    mean_diff_unc = diff_uncertainty.mean().item()

    print(f"Max Probability Absolute Diff : {max_diff_prob:.2e}")
    print(f"Mean Probability Absolute Diff: {mean_diff_prob:.2e}")
    print(f"Max Uncertainty Absolute Diff : {max_diff_unc:.2e}")
    print(f"Mean Uncertainty Absolute Diff: {mean_diff_unc:.2e}")

    # Consistency Loss Equivalence Check
    threshold = 0.520 * np.log(2)
    seq_mask = (seq_uncertainty < threshold).float()
    batched_mask = (batched_uncertainty < threshold).float()
    mask_diff = (seq_mask - batched_mask).abs().sum().item()

    dummy_pred = torch.randn(B_ul, C, H, W)
    dummy_tea = torch.randn(B_ul, C, H, W)
    cons_dist = sigmoid_mse_loss(dummy_pred, dummy_tea)

    seq_cons_loss = (torch.sum(seq_mask * cons_dist) / (2.0 * torch.sum(seq_mask) + 1e-16)).item()
    batched_cons_loss = (torch.sum(batched_mask * cons_dist) / (2.0 * torch.sum(batched_mask) + 1e-16)).item()
    cons_loss_diff = abs(seq_cons_loss - batched_cons_loss)

    print(f"Mask Mismatched Pixels        : {mask_diff}")
    print(f"Consistency Loss Abs Diff     : {cons_loss_diff:.2e}")

    is_equiv = max_diff_prob < 1e-5 and max_diff_unc < 1e-5 and mask_diff == 0 and cons_loss_diff < 1e-5
    print(f"Numerical Equivalence Status  : {'[PASS]' if is_equiv else '[FAIL]'}")

    return {
        "max_diff_prob": max_diff_prob,
        "mean_diff_prob": mean_diff_prob,
        "max_diff_unc": max_diff_unc,
        "mean_diff_unc": mean_diff_unc,
        "mask_diff": mask_diff,
        "cons_loss_diff": cons_loss_diff,
        "is_equiv": is_equiv,
    }


def run_component_benchmarks(num_batches_bench: int = 5):
    """
    Executes controlled micro-benchmarks measuring:
    1. Baseline configuration (Sequential MC, Disk loading, Standard EMA)
    2. Batched MC only
    3. Batched MC + RAM Cache
    4. Batched MC + RAM Cache + In-place Vectorized EMA (Fully Optimized)
    """
    print("\n" + "=" * 70)
    print(f"RUNNING CONTROLLED COMPONENT BENCHMARK ({num_batches_bench} BATCHES)")
    print("=" * 70)

    with open(base_dir / "configs" / "mlua_default.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    torch.set_num_threads(8)

    train_img_dir = base_dir / cfg["data"]["train_image_dir"]
    train_lbl_dir = base_dir / cfg["data"]["train_label_dir"]
    train_img_files = sorted(list(train_img_dir.glob("*.png")), key=lambda x: int(x.stem))
    train_lbl_files = sorted(list(train_lbl_dir.glob("*.png")), key=lambda x: int(x.stem))

    labeled_count = cfg["data"]["labeled_rates"]["0.1"]
    all_indices = list(range(len(train_img_files)))
    labeled_indices = all_indices[:labeled_count]
    unlabeled_indices = all_indices[labeled_count:]

    batch_sampler = TwoStreamBatchSampler(
        labeled_indices,
        unlabeled_indices,
        batch_size=cfg["data"]["batch_size"],
        l_batch_size=cfg["data"]["labeled_batch_size"],
    )

    # 1. Dataset Configurations
    dataset_disk = CachedTrainDataset(train_img_files, train_lbl_files, transize=384, enable_cache=False)
    loader_disk = DataLoader(dataset_disk, batch_sampler=batch_sampler, num_workers=0)

    dataset_ram = CachedTrainDataset(train_img_files, train_lbl_files, transize=384, enable_cache=True)
    loader_ram = DataLoader(dataset_ram, batch_sampler=batch_sampler, num_workers=0)

    # -------------------------------------------------------------
    # BENCHMARK FUNCTION
    # -------------------------------------------------------------
    def benchmark_pipeline(loader, use_batched_mc: bool, use_inplace_ema: bool, name: str):
        seed_everything(42)
        model_stu = Net(in_c=1, out_c=1, encoder_weights=None).to(device)
        model_tea = Net(in_c=1, out_c=1, encoder_weights=None).to(device)
        model_stu.train()
        model_tea.eval()
        for p in model_tea.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(model_stu.parameters(), lr=0.001, weight_decay=0.01)
        bce_loss_fn = F.binary_cross_entropy_with_logits
        dice_loss_fn = DiceLoss()

        timings = {
            "data_load": [],
            "student_forward": [],
            "teacher_mc": [],
            "backward": [],
            "ema": [],
            "total_batch": [],
        }

        mc_t = 8
        l_batch_size = 4
        stride = 4
        C, H, W = 1, 384, 384

        t_data_start = time.time()
        for b_idx, (imgs, gts) in enumerate(loader):
            if b_idx >= num_batches_bench:
                break
            t_data_end = time.time()
            data_time = t_data_end - t_data_start
            t_batch_start = time.time()

            imgs = imgs.unsqueeze(1) if imgs.ndim == 3 else imgs
            gts = gts.unsqueeze(1) if gts.ndim == 3 else gts
            imgs = imgs.to(device)
            gts = gts.to(device)

            # Student forward
            t0 = time.time()
            pred_fused, pred_aux_list = model_stu(imgs)
            t_stu_fwd = time.time() - t0

            # Teacher MC
            t0 = time.time()
            ul_data = imgs[l_batch_size:]
            noise_base = torch.clamp(torch.randn_like(ul_data) * 0.01, -0.1, 0.1)
            with torch.no_grad():
                ul_pred_tea = model_tea(ul_data + noise_base)[0]

            if not use_batched_mc:
                # Sequential MC
                seq_preds = torch.zeros([mc_t * 5 * stride, C, H, W], device=device)
                with torch.no_grad():
                    for i in range(mc_t):
                        ema_in = ul_data + torch.clamp(torch.randn_like(ul_data) * 0.01, -0.1, 0.1)
                        f_p, p_list = model_tea(ema_in)
                        seq_preds[20 * i : 20 * i + 4] = f_p
                        py_p = torch.cat(p_list, dim=0)
                        seq_preds[20 * i + 4 : 20 * i + 20] = py_p
                    seq_preds = seq_preds.reshape(5 * mc_t, stride, C, H, W)
                    mean_preds = torch.mean(seq_preds, dim=0).sigmoid()
                    uncertainty = -2.0 * torch.sum(mean_preds * torch.log(mean_preds + 1e-6), dim=1, keepdim=True)
            else:
                # Batched MC
                noises = torch.clamp(torch.randn(mc_t, stride, C, H, W, device=device) * 0.01, -0.1, 0.1)
                batched_in = (ul_data.unsqueeze(0) + noises).view(mc_t * stride, C, H, W)
                with torch.no_grad():
                    b_final, b_pyramid_list = model_tea(batched_in)
                    b_final_5 = b_final.view(mc_t, stride, C, H, W).unsqueeze(0)
                    b_pyr_5 = torch.stack([h.view(mc_t, stride, C, H, W) for h in b_pyramid_list], dim=0)
                    all_5 = torch.cat([b_final_5, b_pyr_5], dim=0).view(5 * mc_t, stride, C, H, W)
                    mean_preds = torch.mean(all_5, dim=0).sigmoid()
                    uncertainty = -2.0 * torch.sum(mean_preds * torch.log(mean_preds + 1e-6), dim=1, keepdim=True)

            mask = (uncertainty < (0.520 * np.log(2))).float()
            cons_dist = sigmoid_mse_loss(pred_fused[l_batch_size:], ul_pred_tea)
            cons_loss = torch.sum(mask * cons_dist) / (2.0 * torch.sum(mask) + 1e-16)
            t_tea_mc = time.time() - t0

            # Supervised Loss & Backward
            t0 = time.time()
            bce_l = 0.0
            dice_l = 0.0
            for pred_aux in pred_aux_list:
                bce_l += bce_loss_fn(pred_aux[:l_batch_size], gts[:l_batch_size])
                dice_l += dice_loss_fn(pred_aux[:l_batch_size], gts[:l_batch_size])
            bce_l += bce_loss_fn(pred_fused[:l_batch_size], gts[:l_batch_size])
            dice_l += dice_loss_fn(pred_fused[:l_batch_size], gts[:l_batch_size])
            total_loss = 0.5 * (bce_l / 4.0 + dice_l / 4.0) + 0.001 * cons_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            t_backward = time.time() - t0

            # EMA
            t0 = time.time()
            alpha = 0.99
            if not use_inplace_ema:
                for p_tea, p_stu in zip(model_tea.parameters(), model_stu.parameters()):
                    p_tea.data = alpha * p_tea.data + (1.0 - alpha) * p_stu.data
            else:
                for p_tea, p_stu in zip(model_tea.parameters(), model_stu.parameters()):
                    p_tea.data.mul_(alpha).add_(p_stu.data, alpha=1.0 - alpha)
            t_ema = time.time() - t0

            total_b = time.time() - t_batch_start

            timings["data_load"].append(data_time)
            timings["student_forward"].append(t_stu_fwd)
            timings["teacher_mc"].append(t_tea_mc)
            timings["backward"].append(t_backward)
            timings["ema"].append(t_ema)
            timings["total_batch"].append(total_b)

            t_data_start = time.time()

        mean_data = np.mean(timings["data_load"])
        mean_stu = np.mean(timings["student_forward"])
        mean_mc = np.mean(timings["teacher_mc"])
        mean_bwd = np.mean(timings["backward"])
        mean_ema = np.mean(timings["ema"])
        mean_tot = np.mean(timings["total_batch"])

        print(f"\n--- {name} ---")
        print(f"Data Loading Time/Batch : {mean_data:.3f}s")
        print(f"Student Forward Pass    : {mean_stu:.3f}s")
        print(f"Teacher MC Passes (T=8) : {mean_mc:.3f}s")
        print(f"Backward + Loss Step    : {mean_bwd:.3f}s")
        print(f"EMA Update Step         : {mean_ema:.3f}s")
        print(f"Mean Batch Time (Total) : {mean_tot:.3f}s (Projected 66-batch epoch: {mean_tot * 66:.1f}s / {mean_tot * 66 / 60:.2f} min)")

        return {
            "data_load": mean_data,
            "student_forward": mean_stu,
            "teacher_mc": mean_mc,
            "backward": mean_bwd,
            "ema": mean_ema,
            "total_batch": mean_tot,
            "epoch_sec": mean_tot * 66,
        }

    # Execute the 4 benchmark configurations
    res_baseline = benchmark_pipeline(loader_disk, use_batched_mc=False, use_inplace_ema=False, name="1. BASELINE (Sequential MC + Disk IO)")
    res_batched_mc = benchmark_pipeline(loader_disk, use_batched_mc=True, use_inplace_ema=False, name="2. OPTIMIZATION 1: BATCHED MC ONLY")
    res_mc_ram = benchmark_pipeline(loader_ram, use_batched_mc=True, use_inplace_ema=False, name="3. OPTIMIZATION 1+3: BATCHED MC + RAM CACHE")
    res_fully_opt = benchmark_pipeline(loader_ram, use_batched_mc=True, use_inplace_ema=True, name="4. FULL OPTIMIZATION: BATCHED MC + RAM CACHE + INPLACE EMA")

    return {
        "baseline": res_baseline,
        "batched_mc": res_batched_mc,
        "mc_ram": res_mc_ram,
        "fully_opt": res_fully_opt,
    }


if __name__ == "__main__":
    equiv_results = run_numerical_equivalence_test()
    bench_results = run_component_benchmarks(num_batches_bench=6)
    
    out_dir = base_dir / "outputs" / "experiments" / "EXP-MLUA-001" / "compute_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary_data = {
        "numerical_equivalence": equiv_results,
        "benchmarks": bench_results,
    }
    with open(out_dir / "day1_optimization_results.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved optimization results to {out_dir / 'day1_optimization_results.json'}")
