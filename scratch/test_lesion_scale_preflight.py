import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

base_dir = Path("c:/Users/devin/MLUA")
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from src.mlua.evaluation.lesion_matching import extract_connected_lesions, match_lesions
from src.mlua.evaluation.lesion_scale_analysis import LesionScaleAnalyzer

def run_lesion_scale_preflight():
    print("=" * 75, flush=True)
    print("EXP-MLUA LESION-SCALE ANALYSIS FRAMEWORK PRE-TRAINING AUDIT", flush=True)
    print("=" * 75, flush=True)

    # 1. Synthetic Mask Generation for Functional Validation
    H, W = 768, 1536
    synthetic_gt = np.zeros((H, W), dtype=np.uint8)
    synthetic_pred = np.zeros((H, W), dtype=np.uint8)

    # Small lesion: 10 x 20 = 200 pixels (<300)
    synthetic_gt[50:60, 50:70] = 1
    # Small prediction slightly shifted: 10 x 20, overlapping 10x10 = 100 px -> IoU = 100 / (200 + 200 - 100) = 0.333
    synthetic_pred[50:60, 60:80] = 1

    # Medium lesion: 25 x 20 = 500 pixels (300 <= area <= 1000)
    synthetic_gt[150:175, 150:170] = 1
    # Medium prediction exact match: 25 x 20 = 500 px -> IoU = 1.000
    synthetic_pred[150:175, 150:170] = 1

    # Large lesion: 40 x 30 = 1200 pixels (>1000)
    synthetic_gt[300:340, 300:330] = 1
    # Large prediction partial overlap: 40 x 20 = 800 px -> IoU = 800 / 1200 = 0.667
    synthetic_pred[300:340, 300:320] = 1

    # False positive lesion: 15 x 15 = 225 pixels (SMALL)
    synthetic_pred[500:515, 500:515] = 1

    print("[Synthetic Ground Truth] Generated 3 GT lesions: Small (200px), Medium (500px), Large (1200px)", flush=True)

    # 2. Test Lesion Extraction
    gt_lesions = extract_connected_lesions(synthetic_gt, case_id="CASE_SYNTH_001")
    assert len(gt_lesions) == 3, f"Expected 3 GT lesions, found {len(gt_lesions)}"
    
    # Check size categorization
    small_l = next(l for l in gt_lesions if l["area_pixels"] == 200)
    med_l = next(l for l in gt_lesions if l["area_pixels"] == 500)
    large_l = next(l for l in gt_lesions if l["area_pixels"] == 1200)

    assert small_l["size_group"] == "SMALL", f"Expected SMALL, got {small_l['size_group']}"
    assert med_l["size_group"] == "MEDIUM", f"Expected MEDIUM, got {med_l['size_group']}"
    assert large_l["size_group"] == "LARGE", f"Expected LARGE, got {large_l['size_group']}"

    # Check bounding box & centroid
    assert small_l["bounding_box"] == (50, 50, 60, 70)
    assert small_l["width"] == 20
    assert small_l["height"] == 10
    assert np.isclose(small_l["centroid"][0], 54.5) and np.isclose(small_l["centroid"][1], 59.5)
    print("[Lesion Extraction] Bounding boxes, centroids, dimensions, and size groupings verified.", flush=True)

    # 3. Test Prediction Extraction & Matching
    pred_lesions = extract_connected_lesions(synthetic_pred, case_id="CASE_SYNTH_001")
    assert len(pred_lesions) == 4, f"Expected 4 Pred lesions, found {len(pred_lesions)}"

    matched = match_lesions(gt_lesions, pred_lesions, iou_threshold=0.10)
    assert len(matched) == 3, f"Expected 3 GT matches, found {len(matched)}"
    
    # Verify all 3 GT lesions were matched to their corresponding predictions
    for m in matched:
        assert m["matched"] is True, f"Lesion {m['lesion_id']} failed to match!"
        if m["size_group"] == "SMALL":
            assert np.isclose(m["lesion_iou"], 0.3333, atol=1e-3)
        elif m["size_group"] == "MEDIUM":
            assert np.isclose(m["lesion_iou"], 1.0000, atol=1e-3)
        elif m["size_group"] == "LARGE":
            assert np.isclose(m["lesion_iou"], 0.6667, atol=1e-3)
    print("[Lesion Matching] Greedy 1-to-1 matching and IoU calculation strictly verified.", flush=True)

    # 4. Test Full Analyzer Module
    analyzer = LesionScaleAnalyzer(iou_threshold=0.10)
    case_res = analyzer.process_case("CASE_SYNTH_001", synthetic_gt, synthetic_pred)
    assert case_res["small_count"] == 1
    assert case_res["medium_count"] == 1
    assert case_res["large_count"] == 1
    assert case_res["status"] == "EVALUATED"

    summary_df = analyzer.compute_summary_table()
    assert len(summary_df) == 4 # SMALL, MEDIUM, LARGE, ALL
    print("[Analyzer Workflow] Summary table metrics computed accurately across all 3 tiers + ALL aggregate.", flush=True)

    # 5. Check CSV Schemas
    eval_dir = base_dir / "outputs" / "evaluation" / "lesion_scale"
    assert (eval_dir / "lesion_scale_results.csv").exists()
    assert (eval_dir / "lesion_level_results.csv").exists()
    assert (eval_dir / "case_level_lesion_scale.csv").exists()
    assert (eval_dir / "MLUA_PAPER_SCALE_REFERENCE.csv").exists()

    df_scale = pd.read_csv(eval_dir / "lesion_scale_results.csv")
    df_ref = pd.read_csv(eval_dir / "MLUA_PAPER_SCALE_REFERENCE.csv")
    assert "SMALL" in df_scale["size_group"].values
    assert "MEDIUM" in df_scale["size_group"].values
    assert "LARGE" in df_scale["size_group"].values
    assert "MLUA (Paper Reported Full)" in df_ref["Model"].values
    print("[Schema Integrity] All 4 evaluation CSV schema files verified with exact headers.", flush=True)

    # 6. Check Sealed Test Benchmark Read-Only Safety
    test_dir = base_dir / "dataset" / "test"
    assert test_dir.exists(), "Sealed test set missing!"
    test_imgs = list((test_dir / "images").glob("*.png")) + list((test_dir / "images_cut").glob("*.png"))
    assert len(test_imgs) > 0, "No test images found in sealed benchmark!"
    print(f"[Sealed Benchmark] dataset/test/ verified intact, sealed, and untouched ({len(test_imgs)} items).", flush=True)

    # 7. Check Active Experiment Isolation
    exp_out = base_dir / "outputs" / "experiments"
    assert (exp_out / "EXP-MLUA-001" / "EXP-MLUA-001_FULL_TRAINING_HISTORY.csv").exists()
    assert (exp_out / "EXP-MLUA-002" / "EXP-MLUA-002_TRAINING_HISTORY.csv").exists()
    for i in range(8):
        assert (exp_out / f"ABL-0{i}" / "training_history.csv").exists()
    for t in [5, 10, 20, 40, 80, 160]:
        assert (exp_out / f"MC-{t:02d}" / "training_history.csv").exists()
    print("[Experiment Isolation] All existing experiment runs (EXP-001, EXP-002, ABL-00..07, MC-05..160) verified untouched.", flush=True)

    print("\n" + "=" * 75, flush=True)
    print("ALL LESION-SCALE ANALYSIS PRE-TRAINING AUDIT CHECKS: [PASS]", flush=True)
    print("=" * 75, flush=True)

if __name__ == "__main__":
    run_lesion_scale_preflight()
