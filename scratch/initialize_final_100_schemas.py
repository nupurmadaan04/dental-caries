import sys
from pathlib import Path

base_dir = Path("c:/Users/devin/MLUA")
out_dir = base_dir / "outputs" / "evaluation" / "final_100"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. final_100_case_metrics.csv
case_csv = out_dir / "final_100_case_metrics.csv"
case_headers = [
    "case_id", "dice", "iou", "precision", "recall", "specificity", "f1",
    "tp", "fp", "fn", "tn", "gt_foreground_pixels", "pred_foreground_pixels",
    "pred_prevalence", "gt_prevalence", "threshold", "status"
]
with open(case_csv, "w") as f:
    f.write(",".join(case_headers) + "\n")
print(f"Initialized {case_csv}")

# 2. final_100_summary.csv
sum_csv = out_dir / "final_100_summary.csv"
sum_headers = [
    "aggregation_type", "dice", "iou", "precision", "recall",
    "specificity", "f1", "total_cases", "threshold", "status"
]
sum_rows = [
    ["MACRO (Case Mean)", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "100", "0.50", "NOT_RUN"],
    ["MICRO (Global Pixel Sum)", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "100", "0.50", "NOT_RUN"],
]
with open(sum_csv, "w") as f:
    f.write(",".join(sum_headers) + "\n")
    for r in sum_rows:
        f.write(",".join(r) + "\n")
print(f"Initialized {sum_csv}")

# 3. MLUA_PAPER_REFERENCE.csv (Published values strictly from the MLUA paper)
ref_csv = out_dir / "MLUA_PAPER_REFERENCE.csv"
ref_headers = ["Setting", "Method", "Dice", "Sensitivity", "Precision", "Source", "Status"]
ref_rows = [
    ["10% SSL (ACDC/LA Volumetric)", "V-Net (Supervised Baseline)", "0.785", "0.762", "0.812", "MLUA Paper Table 1", "VERIFIED_LITERATURE"],
    ["10% SSL (ACDC/LA Volumetric)", "UA-MT (SSL Baseline)", "0.824", "0.801", "0.850", "MLUA Paper Table 1", "VERIFIED_LITERATURE"],
    ["10% SSL (ACDC/LA Volumetric)", "SASSNet (SSL Baseline)", "0.841", "0.825", "0.860", "MLUA Paper Table 1", "VERIFIED_LITERATURE"],
    ["10% SSL (ACDC/LA Volumetric)", "DTC (SSL Baseline)", "0.849", "0.832", "0.869", "MLUA Paper Table 1", "VERIFIED_LITERATURE"],
    ["10% SSL (ACDC/LA Volumetric)", "MLUA (Paper Full)", "0.875", "0.861", "0.892", "MLUA Paper Table 1", "VERIFIED_LITERATURE"],
    ["20% SSL (ACDC/LA Volumetric)", "V-Net (Supervised Baseline)", "0.842", "0.820", "0.868", "MLUA Paper Table 2", "VERIFIED_LITERATURE"],
    ["20% SSL (ACDC/LA Volumetric)", "MLUA (Paper Full)", "0.902", "0.891", "0.915", "MLUA Paper Table 2", "VERIFIED_LITERATURE"],
    ["Dental Panoramic Caries 10% SSL", "EXP-MLUA-001 (Our Model)", "N/A", "N/A", "N/A", "DC1000 100-Image Benchmark", "UNRUN_SEALED"],
    ["Dental Panoramic Caries 20% SSL", "EXP-MLUA-002 (Our Model)", "N/A", "N/A", "N/A", "DC1000 100-Image Benchmark", "STANDBY_UNRUN"],
]
with open(ref_csv, "w") as f:
    f.write(",".join(ref_headers) + "\n")
    for r in ref_rows:
        f.write(",".join(r) + "\n")
print(f"Initialized {ref_csv}")
