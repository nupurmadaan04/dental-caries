"""
Forensic Read-Only Diagnostic for EXP-MLUA-001 Validation Zero-Dice Investigation
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

base_dir = Path("c:/Users/devin/MLUA")
sys.path.insert(0, str(base_dir))

from src.mlua.models.fpn import Net
from util.utils import DiceLoss

def run_diagnostic():
    device = torch.device("cpu")
    ckpt_path = base_dir / "outputs" / "experiments" / "EXP-MLUA-001" / "checkpoints" / "EXP-MLUA-001_LATEST.pth"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    epoch = ckpt.get("epoch", "unknown")
    print(f"Loaded Checkpoint Epoch: {epoch}")

    model = Net(in_c=1, out_c=1, encoder_weights=None).to(device)
    model.load_state_dict(ckpt["model_stu_state_dict"])
    model.eval()

    # Load raw training patches (without augmentations)
    train_img_dir = base_dir / "data" / "raw" / "DC1000_dataset" / "train" / "images"
    train_lbl_dir = base_dir / "data" / "raw" / "DC1000_dataset" / "train" / "labels"
    img_files = sorted(list(train_img_dir.glob("*.png")), key=lambda x: int(x.stem))
    lbl_files = sorted(list(train_lbl_dir.glob("*.png")), key=lambda x: int(x.stem))

    resize = T.Resize((384, 384))
    to_tensor = T.ToTensor()

    print("\n" + "=" * 70)
    print("DETAILED DIAGNOSTIC FOR SAMPLE #0 (RAW UN-AUGMENTED PATCH)")
    print("=" * 70)

    # Sample 0
    img_raw = Image.open(str(img_files[0])).convert("L")
    lbl_raw = Image.open(str(lbl_files[0])).convert("L")

    img_t = to_tensor(resize(img_raw)).unsqueeze(0).to(device) # [1, 1, 384, 384]
    lbl_t = to_tensor(resize(lbl_raw)).unsqueeze(0).to(device) # [1, 1, 384, 384]
    lbl_bin = (lbl_t > 0.5).float()

    with torch.no_grad():
        pred_fused, pred_aux_list = model(img_t)
        prob = torch.sigmoid(pred_fused)

    print(f"Input Tensor Shape : {img_t.shape}")
    print(f"GT Tensor Shape    : {lbl_t.shape}")
    print(f"Logits Range       : Min = {pred_fused.min().item():.4f}, Max = {pred_fused.max().item():.4f}, Mean = {pred_fused.mean().item():.4f}, Std = {pred_fused.std().item():.4f}")
    print(f"Probability Range  : Min = {prob.min().item():.6f}, Max = {prob.max().item():.6f}, Mean = {prob.mean().item():.6f}, Std = {prob.std().item():.6f}")
    print(f"GT Positive Pixels : {int(lbl_bin.sum().item())} / {384*384} ({lbl_bin.mean().item()*100:.3f}%)")

    # Metrics at 0.5
    pred_05 = (prob > 0.5).float()
    tp_05 = (pred_05 * lbl_bin).sum().item()
    fp_05 = (pred_05 * (1.0 - lbl_bin)).sum().item()
    fn_05 = ((1.0 - pred_05) * lbl_bin).sum().item()
    tn_05 = ((1.0 - pred_05) * (1.0 - lbl_bin)).sum().item()

    eps = 1e-4
    dice_05 = (2.0 * tp_05) / (2.0 * tp_05 + fp_05 + fn_05 + eps)
    iou_05 = tp_05 / (tp_05 + fp_05 + fn_05 + eps)
    prec_05 = tp_05 / (tp_05 + fp_05 + eps)
    rec_05 = tp_05 / (tp_05 + fn_05 + eps)
    spec_05 = tn_05 / (tn_05 + fp_05 + eps)

    print(f"\n--- Metrics at Threshold = 0.5 ---")
    print(f"Predicted Positive Pixels : {int(pred_05.sum().item())}")
    print(f"TP = {int(tp_05)}, FP = {int(fp_05)}, FN = {int(fn_05)}, TN = {int(tn_05)}")
    print(f"Dice = {dice_05:.4f}, IoU = {iou_05:.4f}, Prec = {prec_05:.4f}, Rec = {rec_05:.4f}, Spec = {spec_05:.4f}")

    # Threshold sweep
    thresholds = [0.001, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP ON SAMPLE #0")
    print("=" * 70)
    print(f"{'Thresh':<8} | {'Pred Pos':<10} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Dice':<8} | {'IoU':<8} | {'Precision':<10} | {'Recall':<8}")
    print("-" * 75)
    for th in thresholds:
        p_th = (prob > th).float()
        tp = (p_th * lbl_bin).sum().item()
        fp = (p_th * (1.0 - lbl_bin)).sum().item()
        fn = ((1.0 - p_th) * lbl_bin).sum().item()
        d = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
        j = tp / (tp + fp + fn + eps)
        pr = tp / (tp + fp + eps)
        rc = tp / (tp + fn + eps)
        print(f"{th:<8.3f} | {int(p_th.sum().item()):<10} | {int(tp):<6} | {int(fp):<6} | {int(fn):<6} | {d:<8.4f} | {j:<8.4f} | {pr:<10.4f} | {rc:<8.4f}")

    # Sweep across first 20 labeled samples
    print("\n" + "=" * 70)
    print("DISTRIBUTION ACROSS FIRST 20 LABELED TRAINING SAMPLES (RAW UN-AUGMENTED)")
    print("=" * 70)
    max_probs = []
    mean_probs = []
    gt_pixel_counts = []
    dice_05_list = []
    dice_01_list = []

    for i in range(20):
        img_raw = Image.open(str(img_files[i])).convert("L")
        lbl_raw = Image.open(str(lbl_files[i])).convert("L")

        img_t = to_tensor(resize(img_raw)).unsqueeze(0).to(device)
        lbl_t = to_tensor(resize(lbl_raw)).unsqueeze(0).to(device)
        lbl_bin = (lbl_t > 0.5).float()

        with torch.no_grad():
            p_fused, _ = model(img_t)
            pr = torch.sigmoid(p_fused)

        max_probs.append(pr.max().item())
        mean_probs.append(pr.mean().item())
        gt_pixel_counts.append(int(lbl_bin.sum().item()))

        # Dice at 0.5
        p05 = (pr > 0.5).float()
        tp05 = (p05 * lbl_bin).sum().item()
        fp05 = (p05 * (1.0 - lbl_bin)).sum().item()
        fn05 = ((1.0 - p05) * lbl_bin).sum().item()
        d05 = (2.0 * tp05) / (2.0 * tp05 + fp05 + fn05 + eps)
        dice_05_list.append(d05)

        # Dice at 0.1
        p01 = (pr > 0.1).float()
        tp01 = (p01 * lbl_bin).sum().item()
        fp01 = (p01 * (1.0 - lbl_bin)).sum().item()
        fn01 = ((1.0 - p01) * lbl_bin).sum().item()
        d01 = (2.0 * tp01) / (2.0 * tp01 + fp01 + fn01 + eps)
        dice_01_list.append(d01)

        print(f"Sample #{i:02d} ({img_files[i].name}): GT Pixels = {gt_pixel_counts[-1]:<5} | Prob Min={pr.min().item():.5f}, Max={pr.max().item():.5f}, Mean={pr.mean().item():.5f} | Dice@0.5={d05:.4f}, Dice@0.1={d01:.4f}")

    print("-" * 70)
    print(f"Summary over 20 samples:")
    print(f"Average Max Probability : {np.mean(max_probs):.5f} (Range: {np.min(max_probs):.5f} - {np.max(max_probs):.5f})")
    print(f"Average Mean Probability: {np.mean(mean_probs):.5f}")
    print(f"Average GT Positive Px  : {np.mean(gt_pixel_counts):.1f} px / 147,456 px ({np.mean(gt_pixel_counts)/1474.56:.3f}%)")
    print(f"Mean Dice @ 0.5         : {np.mean(dice_05_list):.4f}")
    print(f"Mean Dice @ 0.1         : {np.mean(dice_01_list):.4f}")

if __name__ == "__main__":
    run_diagnostic()
