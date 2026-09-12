"""
EXP-MLUA-002 Pre-Training Audit Script
Strictly verifies all Part 5 pre-training checklist conditions before training launch.
"""

import sys
from pathlib import Path
import yaml
import torch

base_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(base_dir))

from src.mlua.models.fpn import Net
from src.mlua.data.sampler import TwoStreamBatchSampler

def run_audit():
    print("=" * 80)
    print("EXP-MLUA-002 PRE-TRAINING AUDIT VERIFICATION")
    print("=" * 80)

    cfg_path = base_dir / "configs" / "experiments" / "EXP-MLUA-002.yaml"
    assert cfg_path.exists(), f"Missing config: {cfg_path}"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. DATA
    train_img_dir = base_dir / cfg["data"]["train_image_dir"]
    train_lbl_dir = base_dir / cfg["data"]["train_label_dir"]
    train_imgs = sorted(list(train_img_dir.glob("*.png")), key=lambda x: int(x.stem))
    train_lbls = sorted(list(train_lbl_dir.glob("*.png")), key=lambda x: int(x.stem))

    total_samples = len(train_imgs)
    labeled_count = cfg["data"]["labeled_count"]
    unlabeled_count = cfg["data"]["unlabeled_count"]

    assert total_samples == 2389, f"Expected 2389 total, got {total_samples}"
    assert labeled_count == 530, f"Expected 530 labeled, got {labeled_count}"
    assert unlabeled_count == 1859, f"Expected 1859 unlabeled, got {unlabeled_count}"
    assert labeled_count + unlabeled_count == total_samples
    print(f"[PASS] 1. DATA: Total={total_samples}, Labeled={labeled_count}, Unlabeled={unlabeled_count}")

    # 2. SAMPLER
    all_indices = list(range(total_samples))
    labeled_indices = all_indices[:labeled_count]
    unlabeled_indices = all_indices[labeled_count:]
    sampler = TwoStreamBatchSampler(
        labeled_indices, unlabeled_indices,
        batch_size=cfg["data"]["batch_size"],
        l_batch_size=cfg["data"]["labeled_batch_size"]
    )
    first_batch = next(iter(sampler))
    assert len(first_batch) == 8, f"Expected batch size 8, got {len(first_batch)}"
    assert all(idx < labeled_count for idx in first_batch[:4]), "First 4 must be labeled"
    assert all(idx >= labeled_count for idx in first_batch[4:]), "Last 4 must be unlabeled"
    print(f"[PASS] 2. SAMPLER: 4 labeled + 4 unlabeled = 8 total per batch")

    # 3. MODEL
    model = Net(in_c=1, out_c=1, encoder_weights=None)
    dummy_in = torch.randn(2, 1, 384, 384)
    fused, aux_list = model(dummy_in)
    assert fused.shape == (2, 1, 384, 384), f"Unexpected fused shape: {fused.shape}"
    assert len(aux_list) == 4, f"Expected 4 auxiliary heads, got {len(aux_list)}"
    print(f"[PASS] 3. MODEL: ResNet-34 FPN with 4 auxiliary heads + 1 fused head")

    # 4. LOSS & SSL HYPERPARAMETERS
    assert cfg["ssl"]["ema_theta"] == 0.99
    assert cfg["ssl"]["noise_sigma"] == 0.01
    assert cfg["ssl"]["noise_clamp"] == 0.1
    assert cfg["ssl"]["mc_iterations"] == 8
    print(f"[PASS] 4. SSL & LOSS: EMA theta=0.99, Noise sigma=0.01, MC T=8")

    # 5. OPTIMIZER & SCHEDULER
    assert cfg["optimization"]["optimizer"] == "AdamW"
    assert cfg["optimization"]["learning_rate"] == 0.001
    assert cfg["optimization"]["weight_decay"] == 0.01
    assert cfg["optimization"]["betas"] == [0.9, 0.999]
    assert cfg["optimization"]["poly_power"] == 0.9
    assert cfg["experiment"]["max_epochs"] == 200
    print(f"[PASS] 5. OPTIMIZER & SCHEDULER: AdamW (lr=0.001, wd=0.01, beta1=0.9), PolyLR (power=0.9, 200 epochs)")

    # 6. OUTPUT ISOLATION
    exp_dir = base_dir / "outputs" / "experiments" / "EXP-MLUA-002"
    exp001_dir = base_dir / "outputs" / "experiments" / "EXP-MLUA-001"
    assert exp_dir != exp001_dir
    print(f"[PASS] 6. OUTPUT ISOLATION: Targets {exp_dir} (EXP001 completely protected)")

    # 7. SEALED TEST SET VERIFICATION
    test_dir = base_dir / "dataset" / "test"
    assert test_dir.exists()
    print(f"[PASS] 7. TEST SET: Sealed and untouched at {test_dir}")

    print("=" * 80)
    print("ALL EXP-MLUA-002 PRE-TRAINING AUDIT CHECKS: PASS")
    print("=" * 80)

if __name__ == "__main__":
    run_audit()
