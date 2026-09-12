"""
MLUA Implementation Sanity & Smoke Test Script
Executes non-training validation of data loading, sampling, model forward/backward, and evaluation logic.
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

# Add current workspace to sys.path
base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from src.mlua.data.dataset import TrainDataset, ValDataset
from src.mlua.data.sampler import TwoStreamBatchSampler
from src.mlua.models.fpn import Net
from util.utils import DiceLoss, sigmoid_mse_loss, get_current_consistency_weight
from evaluate.utils import get_data_test_overlap, recompone_overlap, metric_calculate

def run_dataset_verification():
    print("\n" + "="*60)
    print("1. DATASET LOADING VERIFICATION")
    print("="*60)
    train_dir = base_dir / "data" / "raw" / "DC1000_dataset" / "train"
    img_dir = train_dir / "images"
    lbl_dir = train_dir / "labels"

    img_files = sorted(list(img_dir.glob("*.png")), key=lambda x: int(x.stem))
    lbl_files = sorted(list(lbl_dir.glob("*.png")), key=lambda x: int(x.stem))

    print(f"Total training images found: {len(img_files)}")
    print(f"Total training masks found:  {len(lbl_files)}")
    assert len(img_files) == 2389, f"Expected 2389 images, got {len(img_files)}"
    assert len(lbl_files) == 2389, f"Expected 2389 labels, got {len(lbl_files)}"

    img_stems = [f.stem for f in img_files]
    lbl_stems = [f.stem for f in lbl_files]
    assert img_stems == lbl_stems, "Image and mask filenames do not match 1:1!"
    assert len(set(img_stems)) == len(img_stems), "Duplicate filenames detected in images!"
    assert len(set(lbl_stems)) == len(lbl_stems), "Duplicate filenames detected in masks!"

    # Sample check 10 images and masks for shape & mode
    for f in img_files[:10]:
        im = Image.open(f)
        assert im.size == (384, 384), f"Unexpected image size {im.size} in {f.name}"
        assert im.mode == "L", f"Unexpected image mode {im.mode} in {f.name}"

    for f in lbl_files[:10]:
        im = Image.open(f)
        assert im.size == (384, 384), f"Unexpected label size {im.size} in {f.name}"
        arr = np.array(im)
        unique_vals = set(np.unique(arr))
        assert unique_vals.issubset({0, 255}), f"Unexpected mask values {unique_vals} in {f.name}"

    print("[OK] Dataset integrity check: PASSED (2,389 1:1 paired 384x384 grayscale/binary images).")


def run_sampler_verification():
    print("\n" + "="*60)
    print("2. SSL SAMPLER VERIFICATION")
    print("="*60)
    total_samples = 2389
    ratios = {
        "0.1": (265, 2124),
        "0.2": (530, 1859),
        "0.5": (1325, 1064),
    }

    all_idxs = list(range(total_samples))
    for rate, (expected_l, expected_ul) in ratios.items():
        l_idxs = all_idxs[:expected_l]
        ul_idxs = list(set(all_idxs) - set(l_idxs))
        assert len(l_idxs) == expected_l
        assert len(ul_idxs) == expected_ul
        
        sampler = TwoStreamBatchSampler(l_idxs, ul_idxs, batch_size=8, l_batch_size=4)
        expected_batches = expected_l // 4
        assert len(sampler) == expected_batches, f"Expected {expected_batches} batches, got {len(sampler)}"

        # Sample first batch
        batch = next(iter(sampler))
        assert len(batch) == 8, f"Batch size must be 8, got {len(batch)}"
        l_part = batch[:4]
        ul_part = batch[4:]
        assert all(idx < expected_l for idx in l_part), "Labeled batch contains non-labeled indices!"
        assert all(idx >= expected_l for idx in ul_part), "Unlabeled batch contains labeled indices!"
        print(f"[OK] Split {rate} ({expected_l} L / {expected_ul} UL): {expected_batches} batches/epoch verified.")


def run_data_loader_sanity():
    print("\n" + "="*60)
    print("3. REAL DATA LOADER SANITY CHECK (2 BATCHES)")
    print("="*60)
    train_dir = base_dir / "data" / "raw" / "DC1000_dataset" / "train"
    img_files = sorted(list((train_dir / "images").glob("*.png")), key=lambda x: int(x.stem))
    lbl_files = sorted(list((train_dir / "labels").glob("*.png")), key=lambda x: int(x.stem))

    dataset = TrainDataset(img_files, lbl_files)
    l_idxs = list(range(265))
    ul_idxs = list(range(265, 2389))
    sampler = TwoStreamBatchSampler(l_idxs, ul_idxs, batch_size=8, l_batch_size=4)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    for i, (imgs, gts) in enumerate(loader):
        if i >= 2:
            break
        print(f"  Batch {i}: imgs shape={list(imgs.shape)}, gts shape={list(gts.shape)}")
        assert imgs.shape == (8, 384, 384) or imgs.shape == (8, 1, 384, 384)
        assert gts.shape == (8, 384, 384) or gts.shape == (8, 1, 384, 384)
        assert not torch.isnan(imgs).any(), "NaN in image tensor!"
        assert not torch.isinf(imgs).any(), "Inf in image tensor!"
        assert imgs.min() >= 0.0 and imgs.max() <= 1.0, f"Image range out of [0, 1]: [{imgs.min()}, {imgs.max()}]"
        assert gts.min() >= 0.0 and gts.max() <= 1.0, f"Mask range out of [0, 1]: [{gts.min()}, {gts.max()}]"

    print("[OK] Data loader sanity check: PASSED (Loaded real images/masks without training).")


def run_model_smoke_test():
    print("\n" + "="*60)
    print("4. SYNTHETIC MODEL FORWARD/BACKWARD SMOKE TEST")
    print("="*60)
    model = Net(in_c=1, out_c=1, encoder_weights=None)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    bce_loss = F.binary_cross_entropy_with_logits
    dice_loss = DiceLoss()

    # Tiny synthetic input (batch_size = 2, channels = 1, 384x384)
    synthetic_x = torch.randn(2, 1, 384, 384, requires_grad=True)
    synthetic_y = torch.randint(0, 2, (2, 1, 384, 384)).float()

    print(f"  Input synthetic tensor shape: {list(synthetic_x.shape)}")
    fused_mask, aux_masks = model(synthetic_x)

    print(f"  Fused output shape: {list(fused_mask.shape)}")
    print(f"  Auxiliary outputs count: {len(aux_masks)}, shapes: {[list(m.shape) for m in aux_masks]}")
    assert fused_mask.shape == (2, 1, 384, 384), f"Unexpected fused shape {fused_mask.shape}"
    assert len(aux_masks) == 4, f"Expected 4 auxiliary heads, got {len(aux_masks)}"
    for idx, aux in enumerate(aux_masks):
        assert aux.shape == (2, 1, 384, 384), f"Aux {idx} shape mismatch: {aux.shape}"

    # Compute loss
    tot_bce = bce_loss(fused_mask, synthetic_y)
    tot_dice = dice_loss(fused_mask, synthetic_y)
    for aux in aux_masks:
        tot_bce += bce_loss(aux, synthetic_y)
        tot_dice += dice_loss(aux, synthetic_y)

    loss = 0.5 * (tot_bce / 4 + tot_dice / 4)
    print(f"  Calculated smoke loss: {loss.item():.4f}")

    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.parameters())
    assert has_grad, "No gradients computed during backward pass!"
    optimizer.step()

    print("[OK] Model forward/backward smoke test: PASSED (Verified FPN + 4 Aux Heads on synthetic data).")


def run_evaluation_code_verification():
    print("\n" + "="*60)
    print("5. EVALUATION PIPELINE CODE-LEVEL VERIFICATION")
    print("="*60)
    # Synthetic panoramic test (768x1536)
    img_h, img_w = 768, 1536
    patch_h, patch_w = 384, 384
    stride_h, stride_w = 192, 192

    # Expected patches count
    nh = (img_h - patch_h) // stride_h + 1  # 3
    nw = (img_w - patch_w) // stride_w + 1  # 7
    tot_patches = nh * nw  # 21

    print(f"  Sliding window grid: {nh} vertical x {nw} horizontal = {tot_patches} patches per panorama.")
    assert tot_patches == 21, f"Expected 21 patches, got {tot_patches}"

    # Synthetic prediction patches [21, 1, 384, 384]
    synthetic_preds = np.random.uniform(0.0, 1.0, (tot_patches, 1, patch_h, patch_w))
    recomposed = recompone_overlap(synthetic_preds, img_h, img_w, stride_h, stride_w)

    print(f"  Recomposed full panoramic probability map shape: {recomposed.shape}")
    assert recomposed.shape == (1, 1, img_h, img_w), f"Unexpected recomposed shape {recomposed.shape}"
    assert 0.0 <= recomposed.min() and recomposed.max() <= 1.0, "Recomposed values out of [0, 1]!"

    # Test metric_calculate on synthetic binary arrays
    target = (np.random.rand(img_h, img_w) > 0.8).astype(np.uint8)
    prediction = (recomposed.squeeze() > 0.5).astype(np.uint8)
    acc, iou, dice, pre, spe, sen = metric_calculate(target, prediction)
    print(f"  Sample metric_calculate output: Dice={dice:.4f}, IoU={iou:.4f}, Prec={pre:.4f}, Sen={sen:.4f}, Spe={spe:.4f}")
    assert 0.0 <= dice <= 1.0, f"Dice out of bounds: {dice}"

    print("[OK] Evaluation pipeline code verification: PASSED (Zero test cases evaluated).")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("STARTING MLUA STANDALONE IMPLEMENTATION SANITY CHECKS")
    print("="*60)
    run_dataset_verification()
    run_sampler_verification()
    run_data_loader_sanity()
    run_model_smoke_test()
    run_evaluation_code_verification()
    print("\n" + "="*60)
    print("ALL MLUA SANITY CHECKS COMPLETED SUCCESSFULLY!")
    print("="*60)
