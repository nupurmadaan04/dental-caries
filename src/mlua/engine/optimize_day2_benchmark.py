"""
DAY 2: EXP-MLUA-001 Teacher MC Optimization & Verification Script
Compares:
1. REFERENCE: Current Batched-MC (2 separate teacher forward passes: 4 images + 32 images, torch.no_grad)
2. OPTIMIZED: Unified Teacher Forward (1 single 36-image forward pass, torch.inference_mode, pre-allocated buffers)
3. NUMERICAL EQUIVALENCE: Probability, Uncertainty, Mask, Consistency Loss
4. 12-BATCH PERFORMANCE BENCHMARK & MULTI-EPOCH PROJECTIONS
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
        self.nomalize_transform = T.ToTensor()

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
        image = self.cached_images[index].copy()
        label = self.cached_labels[index].copy()

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


def run_day2_numerical_equivalence_test():
    """
    Tests exact numerical equivalence between:
    1. Reference Batched-MC (2 separate passes: 4 images + 32 images)
    2. Optimized Unified 36-image Teacher Forward Pass under inference_mode
    """
    print("\n" + "=" * 70)
    print("DAY 2: NUMERICAL EQUIVALENCE TEST (REFERENCE BATCHED-MC VS UNIFIED 36-IMAGE MC)")
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

    ul_data = torch.randn(B_ul, C, H, W, device=device)
    noise_sigma = 0.01
    noise_clamp = 0.1

    # Fixed identical perturbations
    torch.manual_seed(999)
    noise_base = torch.clamp(torch.randn_like(ul_data) * noise_sigma, -noise_clamp, noise_clamp)
    noises_mc = torch.clamp(torch.randn(T_mc, B_ul, C, H, W, device=device) * noise_sigma, -noise_clamp, noise_clamp)

    # -------------------------------------------------------------
    # 1. REFERENCE BATCHED-MC (Day 1 Implementation)
    # -------------------------------------------------------------
    with torch.no_grad():
        # Pass 1: Base pass
        ref_ul_pred_tea = model_tea(ul_data + noise_base)[0]

        # Pass 2: 32-image MC pass
        ref_batched_in = (ul_data.unsqueeze(0) + noises_mc).view(T_mc * B_ul, C, H, W)
        ref_final, ref_pyramid_list = model_tea(ref_batched_in)
        ref_final_5 = ref_final.view(T_mc, B_ul, C, H, W).unsqueeze(0)
        ref_pyr_5 = torch.stack([h.view(T_mc, B_ul, C, H, W) for h in ref_pyramid_list], dim=0)
        ref_all_5 = torch.cat([ref_final_5, ref_pyr_5], dim=0).view(5 * T_mc, B_ul, C, H, W)
        ref_mean_preds = torch.mean(ref_all_5, dim=0).sigmoid()
        ref_uncertainty = -2.0 * torch.sum(ref_mean_preds * torch.log(ref_mean_preds + 1e-6), dim=1, keepdim=True)

    # -------------------------------------------------------------
    # 2. OPTIMIZED UNIFIED 36-IMAGE FORWARD PASS (Day 2 Candidate)
    # -------------------------------------------------------------
    with torch.inference_mode():
        # Pre-concatenate into single 36-image tensor: [4 + 32, 1, 384, 384]
        opt_all_in = torch.empty((B_ul + T_mc * B_ul, C, H, W), device=device)
        opt_all_in[:B_ul] = ul_data + noise_base
        opt_all_in[B_ul:] = (ul_data.unsqueeze(0) + noises_mc).view(T_mc * B_ul, C, H, W)

        # Single forward call for all 36 teacher evaluations
        opt_final_all, opt_pyramid_all = model_tea(opt_all_in)

        # Split consistency target and MC evaluations
        opt_ul_pred_tea = opt_final_all[:B_ul]
        opt_final_mc = opt_final_all[B_ul:]
        opt_pyr_mc = [h[B_ul:] for h in opt_pyramid_all]

        opt_final_5 = opt_final_mc.view(T_mc, B_ul, C, H, W).unsqueeze(0)
        opt_pyr_5 = torch.stack([h.view(T_mc, B_ul, C, H, W) for h in opt_pyr_mc], dim=0)
        opt_all_5 = torch.cat([opt_final_5, opt_pyr_5], dim=0).view(5 * T_mc, B_ul, C, H, W)
        opt_mean_preds = torch.mean(opt_all_5, dim=0).sigmoid()
        opt_uncertainty = -2.0 * torch.sum(opt_mean_preds * torch.log(opt_mean_preds + 1e-6), dim=1, keepdim=True)

    # -------------------------------------------------------------
    # 3. NUMERICAL EQUIVALENCE EVALUATION
    # -------------------------------------------------------------
    diff_target = (ref_ul_pred_tea - opt_ul_pred_tea).abs()
    diff_prob = (ref_mean_preds - opt_mean_preds).abs()
    diff_unc = (ref_uncertainty - opt_uncertainty).abs()

    max_diff_target = diff_target.max().item()
    max_diff_prob = diff_prob.max().item()
    mean_diff_prob = diff_prob.mean().item()
    max_diff_unc = diff_unc.max().item()
    mean_diff_unc = diff_unc.mean().item()

    threshold = 0.520 * np.log(2)
    ref_mask = (ref_uncertainty < threshold).float()
    opt_mask = (opt_uncertainty < threshold).float()
    mask_diff = (ref_mask - opt_mask).abs().sum().item()

    dummy_pred = torch.randn(B_ul, C, H, W)
    ref_cons_dist = sigmoid_mse_loss(dummy_pred, ref_ul_pred_tea)
    opt_cons_dist = sigmoid_mse_loss(dummy_pred, opt_ul_pred_tea)

    ref_cons_loss = (torch.sum(ref_mask * ref_cons_dist) / (2.0 * torch.sum(ref_mask) + 1e-16)).item()
    opt_cons_loss = (torch.sum(opt_mask * opt_cons_dist) / (2.0 * torch.sum(opt_mask) + 1e-16)).item()
    cons_loss_diff = abs(ref_cons_loss - opt_cons_loss)

    print(f"Max Consistency Target Diff   : {max_diff_target:.2e}")
    print(f"Max Probability Absolute Diff : {max_diff_prob:.2e}")
    print(f"Mean Probability Absolute Diff: {mean_diff_prob:.2e}")
    print(f"Max Uncertainty Absolute Diff : {max_diff_unc:.2e}")
    print(f"Mean Uncertainty Absolute Diff: {mean_diff_unc:.2e}")
    print(f"Mask Mismatched Pixels        : {mask_diff}")
    print(f"Consistency Loss Abs Diff     : {cons_loss_diff:.2e}")

    is_equiv = (
        max_diff_target < 1e-5
        and max_diff_prob < 1e-5
        and max_diff_unc < 1e-5
        and mask_diff == 0
        and cons_loss_diff < 1e-6
    )
    print(f"Day 2 Numerical Equivalence   : {'[PASS]' if is_equiv else '[FAIL]'}")

    return {
        "max_diff_target": max_diff_target,
        "max_diff_prob": max_diff_prob,
        "mean_diff_prob": mean_diff_prob,
        "max_diff_unc": max_diff_unc,
        "mean_diff_unc": mean_diff_unc,
        "mask_diff": mask_diff,
        "cons_loss_diff": cons_loss_diff,
        "is_equiv": is_equiv,
        "ref_mean_prob": ref_mean_preds.mean().item(),
        "opt_mean_prob": opt_mean_preds.mean().item(),
        "ref_mean_unc": ref_uncertainty.mean().item(),
        "opt_mean_unc": opt_uncertainty.mean().item(),
    }


def run_day2_benchmark(num_measurement_batches: int = 12):
    """
    Executes 12-batch comparative micro-benchmark:
    1. Reference: Current Batched-MC (2 separate forward passes: 4 images + 32 images)
    2. Optimized: Unified 36-image Teacher forward pass (1 forward pass, torch.inference_mode)
    """
    print("\n" + "=" * 70)
    print(f"DAY 2: 12-BATCH PERFORMANCE BENCHMARK ({num_measurement_batches} BATCHES)")
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

    dataset_ram = CachedTrainDataset(train_img_files, train_lbl_files, transize=384)
    loader = DataLoader(dataset_ram, batch_sampler=batch_sampler, num_workers=0)

    def benchmark_mode(use_unified_36: bool, name: str):
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
            if b_idx >= num_measurement_batches:
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

            if not use_unified_36:
                # REFERENCE: 2 separate calls
                with torch.no_grad():
                    ul_pred_tea = model_tea(ul_data + noise_base)[0]
                    noises = torch.clamp(torch.randn(mc_t, stride, C, H, W, device=device) * 0.01, -0.1, 0.1)
                    batched_in = (ul_data.unsqueeze(0) + noises).view(mc_t * stride, C, H, W)
                    b_final, b_pyramid_list = model_tea(batched_in)
                    b_final_5 = b_final.view(mc_t, stride, C, H, W).unsqueeze(0)
                    b_pyr_5 = torch.stack([h.view(mc_t, stride, C, H, W) for h in b_pyramid_list], dim=0)
                    all_5 = torch.cat([b_final_5, b_pyr_5], dim=0).view(5 * mc_t, stride, C, H, W)
                    mean_preds = torch.mean(all_5, dim=0).sigmoid()
                    uncertainty = -2.0 * torch.sum(mean_preds * torch.log(mean_preds + 1e-6), dim=1, keepdim=True)
            else:
                # OPTIMIZED: 1 unified 36-image call with inference_mode
                with torch.inference_mode():
                    noises = torch.clamp(torch.randn(mc_t, stride, C, H, W, device=device) * 0.01, -0.1, 0.1)
                    unified_in = torch.empty((stride + mc_t * stride, C, H, W), device=device)
                    unified_in[:stride] = ul_data + noise_base
                    unified_in[stride:] = (ul_data.unsqueeze(0) + noises).view(mc_t * stride, C, H, W)

                    all_final, all_pyramid = model_tea(unified_in)
                    ul_pred_tea = all_final[:stride]
                    mc_final = all_final[stride:]
                    mc_pyr = [h[stride:] for h in all_pyramid]

                    b_final_5 = mc_final.view(mc_t, stride, C, H, W).unsqueeze(0)
                    b_pyr_5 = torch.stack([h.view(mc_t, stride, C, H, W) for h in mc_pyr], dim=0)
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

            # EMA in-place
            t0 = time.time()
            alpha = 0.99
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

        mean_data = float(np.mean(timings["data_load"]))
        mean_stu = float(np.mean(timings["student_forward"]))
        mean_mc = float(np.mean(timings["teacher_mc"]))
        mean_bwd = float(np.mean(timings["backward"]))
        mean_ema = float(np.mean(timings["ema"]))
        mean_tot = float(np.mean(timings["total_batch"]))
        epoch_sec = mean_tot * 66

        print(f"\n--- {name} ({num_measurement_batches} Batches) ---")
        print(f"Data Loading Time/Batch : {mean_data:.3f}s")
        print(f"Student Forward Pass    : {mean_stu:.3f}s")
        print(f"Teacher MC Passes (T=8) : {mean_mc:.3f}s")
        print(f"Backward + Loss Step    : {mean_bwd:.3f}s")
        print(f"EMA Update Step         : {mean_ema:.3f}s")
        print(f"Mean Batch Time (Total) : {mean_tot:.3f}s")
        print(f"Projected Epoch Runtime : {epoch_sec:.1f}s ({epoch_sec / 60:.2f} min)")

        return {
            "data_load": mean_data,
            "student_forward": mean_stu,
            "teacher_mc": mean_mc,
            "backward": mean_bwd,
            "ema": mean_ema,
            "total_batch": mean_tot,
            "epoch_sec": epoch_sec,
            "epoch_min": epoch_sec / 60.0,
            "runtime_25_epochs_hr": (epoch_sec * 25) / 3600.0,
            "runtime_30_epochs_hr": (epoch_sec * 30) / 3600.0,
            "runtime_40_epochs_hr": (epoch_sec * 40) / 3600.0,
            "runtime_200_epochs_hr": (epoch_sec * 200) / 3600.0,
        }

    res_ref = benchmark_mode(use_unified_36=False, name="1. REFERENCE BATCHED-MC (2 Forward Calls)")
    res_opt = benchmark_mode(use_unified_36=True, name="2. OPTIMIZED UNIFIED 36-IMAGE MC (1 Forward Call + InferenceMode)")

    return {
        "reference": res_ref,
        "optimized": res_opt,
    }


if __name__ == "__main__":
    equiv_results = run_day2_numerical_equivalence_test()
    bench_results = run_day2_benchmark(num_measurement_batches=12)

    out_dir = base_dir / "outputs" / "experiments" / "EXP-MLUA-001" / "compute_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "numerical_equivalence": equiv_results,
        "benchmark_12_batches": bench_results,
    }

    with open(out_dir / "day2_mc_optimization_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved Day 2 optimization results to {out_dir / 'day2_mc_optimization_results.json'}")
