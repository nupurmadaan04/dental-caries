# MLUA Research & Historical Artifact Archive

This directory houses historical research scripts, exploratory diagnostics, forensic investigations, and reference documentation preserved for scientific traceability and reproducibility without cluttering production execution paths.

## Directory Structure

```
research_archive/
├── reports/
│   ├── 1-s2.0-S0925231223003193-main.pdf            # Foundation literature reference paper
│   └── Detailed_Capstone_Project_Progress_Report.pdf # Milestone capstone engineering report
└── scripts/
    ├── audit_exp001_vs_exp002.py                    # Forensic comparative audit
    ├── build_academic_pdf.py                        # Automated academic paper generator
    ├── build_capstone_progress_pdf.py               # Capstone progress reporting generator
    ├── build_detailed_capstone_progress_pdf.py      # Detailed engineering progress documentation
    ├── check_test_dimensions.py                     # Image dimension and tensor verification
    ├── evaluate_final_100_exp003.py                 # Initial evaluation script for EXP-MLUA-003
    ├── generate_ablation_configs.py                 # Automated configuration generator for ABL-00..07
    ├── generate_mc_configs.py                       # Automated configuration generator for MC-05..160
    ├── generate_progress_figures.py                 # Training curve and telemetry plotting
    ├── generate_report_figures.py                   # High-resolution clinical figures for report
    ├── pretraining_gate_check.py                    # Pre-training validation assertions
    ├── profile_compute.py                           # Hardware profiling and GPU memory benchmarks
    ├── run_threshold_sensitivity.py                 # Operating threshold sweep analysis
    └── ... (additional forensic, validation, and historical analysis scripts)
```

## Reproducibility Note
Active production inference and evaluation are canonically located in:
- Backend AI Service: `backend/`
- Frontend UI: `frontend/`
- Production MLUA Engine: `src/mlua/`
- Active Model Checkpoint: `outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E75_BEST.pth`
