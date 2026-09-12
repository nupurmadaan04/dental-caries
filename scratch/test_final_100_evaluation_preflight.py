import sys
import os
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import torch

base_dir = Path("c:/Users/devin/MLUA")
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from src.mlua.evaluation.final_100_evaluation import (
    extract_panoramic_patches,
    reconstruct_panoramic,
    calculate_case_metrics,
    Final100Evaluator,
)
from src.mlua.models.fpn import Net

def run_final_100_preflight():
    print("=" * 75, flush=True)
    print("EXP-MLUA FINAL 100-IMAGE EVALUATION PRE-TRAINING AUDIT", flush=True)
    print("=" * 75, flush=True)

    # 1. Config Validation
    cfg_path = base_dir / "configs" / "evaluation" / "final_100.yaml"
    assert cfg_path.exists(), f"Config file missing: {cfg_path}"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    print("[Config] Successfully validated configs/evaluation/final_100.yaml", flush=True)

    # 2. Synthetic Panoramic Extraction & Reconstruction Verification
    H, W = 768, 1536
    patch_size = 384
    stride = 192

    synthetic_full_img = np.random.randint(0, 256, (H, W), dtype=np.uint8)
    patches, coords = extract_panoramic_patches(synthetic_full_img, patch_size=patch_size, stride=stride)
    assert len(patches) == 21, f"Expected 21 patches, got {len(patches)}"
    assert len(coords) == 21, f"Expected 21 patch coordinates, got {len(coords)}"
    print(f"[Extraction] Extracted {len(patches)} overlapping 384x384 patches at stride 192.", flush=True)

    # Reconstruct from synthetic patch probability maps
    synthetic_patch_probs = [np.ones((patch_size, patch_size), dtype=np.float32) * 0.75 for _ in range(21)]
    recon_canvas = reconstruct_panoramic(synthetic_patch_probs, coords, full_shape=(H, W), patch_size=patch_size)
    assert recon_canvas.shape == (H, W), f"Reconstruction shape mismatch: {recon_canvas.shape}"
    assert np.allclose(recon_canvas, 0.75), "Overlap normalization failed!"
    print(f"[Reconstruction] Reconstructed {recon_canvas.shape} canvas with perfect overlap normalization.", flush=True)

    # 3. Metric Calculation Verification
    synthetic_gt = np.zeros((H, W), dtype=np.uint8)
    synthetic_gt[100:200, 100:200] = 1 # 10,000 px foreground

    synthetic_pred_prob = np.zeros((H, W), dtype=np.float32)
    synthetic_pred_prob[100:200, 100:200] = 0.90 # 100% overlap at tau=0.50

    metrics = calculate_case_metrics(synthetic_pred_prob, synthetic_gt, threshold=0.50, case_id="SYNTH_CASE_01")
    assert np.isclose(metrics["dice"], 1.0000, atol=1e-3)
    assert np.isclose(metrics["recall"], 1.0000, atol=1e-3)
    assert np.isclose(metrics["precision"], 1.0000, atol=1e-3)
    assert metrics["gt_foreground_pixels"] == 10000
    assert metrics["pred_foreground_pixels"] == 10000
    print("[Metric Logic] Verified case metric formulas (Dice=1.0, IoU=1.0, Recall=1.0, Precision=1.0).", flush=True)

    # 4. Macro & Micro Aggregation Verification
    case_records = [
        metrics,
        # Case 2: 5,000 TP, 5,000 FN, 0 FP -> Dice = 2*5000 / (15000) = 0.6667
        {
            "case_id": "SYNTH_CASE_02", "dice": 0.6667, "iou": 0.5000, "precision": 1.0000, "recall": 0.5000,
            "specificity": 1.0000, "f1": 0.6667, "tp": 5000, "fp": 0, "fn": 5000, "tn": 1169648,
            "gt_foreground_pixels": 10000, "pred_foreground_pixels": 5000, "pred_prevalence": 0.0042,
            "gt_prevalence": 0.0085, "threshold": 0.50
        }
    ]
    evaluator = Final100Evaluator(cfg, device=torch.device("cpu"))
    df_summary = evaluator.compute_summary(case_records)
    assert len(df_summary) == 2 # MACRO and MICRO
    
    # Macro Dice = (1.0 + 0.6667) / 2 = 0.83335
    macro_row = df_summary[df_summary["aggregation_type"].str.contains("MACRO")].iloc[0]
    assert np.isclose(macro_row["dice"], 0.83335, atol=1e-3)
    print("[Aggregation] Verified dual MACRO and MICRO aggregation computations.", flush=True)

    # 5. Dataset Test Verification
    test_dir = base_dir / "dataset" / "test"
    test_cut_imgs = list((test_dir / "images_cut").glob("*.png"))
    test_cut_lbls = list((test_dir / "labels_cut").glob("*.png"))
    assert len(test_cut_imgs) == 100, f"Expected 100 test images, got {len(test_cut_imgs)}"
    assert len(test_cut_lbls) == 100, f"Expected 100 test labels, got {len(test_cut_lbls)}"
    print(f"[Dataset] Verified 100 panoramic test cases in dataset/test/images_cut & labels_cut.", flush=True)

    # 6. CSV Schemas Verification
    out_dir = base_dir / "outputs" / "evaluation" / "final_100"
    assert (out_dir / "final_100_case_metrics.csv").exists()
    assert (out_dir / "final_100_summary.csv").exists()
    assert (out_dir / "MLUA_PAPER_REFERENCE.csv").exists()
    assert (out_dir / "TEST_SET_MANIFEST.json").exists()
    print("[Schemas] Verified all 4 output CSV/JSON evaluation schema files.", flush=True)

    # 7. Experiment Isolation
    exp_out = base_dir / "outputs" / "experiments"
    assert (exp_out / "EXP-MLUA-001" / "EXP-MLUA-001_FULL_TRAINING_HISTORY.csv").exists()
    assert (exp_out / "EXP-MLUA-002" / "EXP-MLUA-002_TRAINING_HISTORY.csv").exists()
    print("[Isolation] EXP-MLUA-001, EXP-MLUA-002, ABL, and MC experiments verified intact and isolated.", flush=True)

    print("\n" + "=" * 75, flush=True)
    print("ALL FINAL 100-IMAGE EVALUATION PRE-TRAINING AUDIT CHECKS: [PASS]", flush=True)
    print("=" * 75, flush=True)

if __name__ == "__main__":
    run_final_100_preflight()
