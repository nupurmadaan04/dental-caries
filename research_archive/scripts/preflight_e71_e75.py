import os
import sys
import hashlib
import torch
import yaml
from pathlib import Path

base_dir = Path(r"c:\Users\devin\MLUA")
sys.path.insert(0, str(base_dir))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.mlua.models.fpn import Net
from src.mlua.data.sampler import TwoStreamBatchSampler

def run_preflight():
    print("=== EXP-MLUA-003: PREFLIGHT RESUME VALIDATION (E70 -> E71) ===")
    
    # 1. Check E64 SHA256
    e64_path = base_dir / "outputs" / "experiments" / "EXP-MLUA-003_FINAL" / "checkpoints" / "EXP-MLUA-003_E64_BEST.pth"
    assert e64_path.exists(), f"E64 missing at {e64_path}"
    with open(e64_path, "rb") as f:
        e64_hash = hashlib.sha256(f.read()).hexdigest()
    expected_e64_hash = "167918b8375815668b5e784902e3e82a865b86053a610a312c775d52d473c526"
    print(f"E64 Checkpoint SHA256: {e64_hash}")
    assert e64_hash == expected_e64_hash, f"E64 hash mismatch! Expected {expected_e64_hash}, got {e64_hash}"
    print("✓ E64 Checkpoint verified and frozen.")

    # 2. Check E70 Checkpoint
    e70_path = base_dir / "outputs" / "experiments" / "EXP-MLUA-003_FINAL" / "checkpoints" / "EXP-MLUA-003_E70_LATEST.pth"
    assert e70_path.exists(), f"E70 missing at {e70_path}"
    print(f"Loading E70 checkpoint from: {e70_path}")
    ckpt = torch.load(e70_path, map_location="cpu", weights_only=False)
    
    required_keys = [
        "epoch", "model_stu_state_dict", "model_tea_state_dict",
        "optimizer_state_dict", "scheduler_state_dict", "global_step", "metrics", "config"
    ]
    for k in required_keys:
        assert k in ckpt, f"Required continuation key '{k}' missing from E70 checkpoint!"
    
    epoch = ckpt["epoch"]
    global_step = ckpt["global_step"]
    print(f"✓ Checkpoint keys verified. Epoch: {epoch}, Global Step: {global_step}")
    assert epoch == 70, f"Expected checkpoint epoch 70, got {epoch}"
    assert global_step == 9240, f"Expected global step 9240 (70 * 132), got {global_step}"

    # 3. Model Architecture & State Dict loading
    device = torch.device("cpu")
    model_stu = Net(in_c=1, out_c=1, encoder_weights=None).to(device)
    model_tea = Net(in_c=1, out_c=1, encoder_weights=None).to(device)
    
    model_stu.load_state_dict(ckpt["model_stu_state_dict"])
    model_tea.load_state_dict(ckpt["model_tea_state_dict"])
    print("✓ Model student and teacher state_dicts loaded successfully.")

    # Check finite weights
    for name, p in model_stu.named_parameters():
        assert torch.isfinite(p).all(), f"Nonfinite param in student: {name}"
    for name, p in model_tea.named_parameters():
        assert torch.isfinite(p).all(), f"Nonfinite param in teacher: {name}"
    print("✓ All student and teacher parameters are 100% FINITE.")

    # Check Teacher BatchNorm buffers
    tea_buffers = dict(model_tea.named_buffers())
    assert len(tea_buffers) > 0, "No buffers in teacher model!"
    for name, b in tea_buffers.items():
        assert torch.isfinite(b).all(), f"Nonfinite buffer in teacher: {name}"
    print(f"✓ All {len(tea_buffers)} teacher buffers are 100% FINITE.")

    # 4. Optimizer and Scheduler State
    cfg = ckpt["config"]
    lr = cfg["optimization"]["learning_rate"]
    weight_decay = cfg["optimization"]["weight_decay"]
    max_epochs = cfg["experiment"]["max_epochs"]
    poly_lr_fn = lambda ep: (1.0 - float(ep) / max_epochs) ** cfg["optimization"]["poly_power"]

    optimizer = torch.optim.AdamW(model_stu.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr_fn)

    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print("✓ Optimizer and LambdaLR scheduler state_dicts loaded successfully.")

    curr_lr = optimizer.param_groups[0]["lr"]
    print(f"✓ Current Learning Rate: {curr_lr:.7f}")
    assert curr_lr > 0.0, "Learning rate is not positive!"

    # 5. Dataset Verification
    train_img_dir = base_dir / cfg["data"]["train_image_dir"]
    train_lbl_dir = base_dir / cfg["data"]["train_label_dir"]
    train_img_files = sorted(list(train_img_dir.glob("*.png")), key=lambda x: int(x.stem))
    train_lbl_files = sorted(list(train_lbl_dir.glob("*.png")), key=lambda x: int(x.stem))
    
    assert len(train_img_files) == 2389, f"Expected 2389 images, got {len(train_img_files)}"
    assert len(train_lbl_files) == 2389, f"Expected 2389 labels, got {len(train_lbl_files)}"
    print(f"✓ DC1000 dataset confirmed: 2389 patches (530 labeled, 1859 unlabeled).")

    print("\n>>> ALL PREFLIGHT CHECKS PASSED. SYSTEM READY FOR E71 CONTINUATION. <<<")

if __name__ == "__main__":
    run_preflight()
