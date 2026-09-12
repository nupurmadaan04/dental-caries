import os
import yaml
from pathlib import Path

base_dir = Path("c:/Users/devin/MLUA")
mc_configs_dir = base_dir / "configs" / "experiments" / "mc_sampling"
mc_configs_dir.mkdir(parents=True, exist_ok=True)

mc_experiments = [
    {"id": "MC-05", "samples": 5, "desc": "Monte Carlo Sampling Sensitivity Study: T=5 stochastic noisy passes"},
    {"id": "MC-10", "samples": 10, "desc": "Monte Carlo Sampling Sensitivity Study: T=10 stochastic noisy passes"},
    {"id": "MC-20", "samples": 20, "desc": "Monte Carlo Sampling Sensitivity Study: T=20 stochastic noisy passes"},
    {"id": "MC-40", "samples": 40, "desc": "Monte Carlo Sampling Sensitivity Study: T=40 stochastic noisy passes"},
    {"id": "MC-80", "samples": 80, "desc": "Monte Carlo Sampling Sensitivity Study: T=80 stochastic noisy passes"},
    {"id": "MC-160", "samples": 160, "desc": "Monte Carlo Sampling Sensitivity Study: T=160 stochastic noisy passes"},
]

for exp in mc_experiments:
    exp_id = exp["id"]
    t_val = exp["samples"]
    cfg = {
        "experiment": {
            "experiment_id": exp_id,
            "name": f"{exp_id}_T{t_val}",
            "description": exp["desc"],
            "seed": 42,
            "ssl_enabled": True,
            "max_epochs": 200,
            "precision": 16,
        },
        "mc_sampling": {
            "mc_samples": t_val,
            "base_evaluations": 1,
            "total_unlabeled_evaluations": 1 + t_val,
            "multiscale_heads_evaluated": 5,
            "total_ensemble_maps": 5 * t_val,
        },
        "model": {
            "name": "Net",
            "encoder": "resnet34",
            "encoder_weights": "imagenet",
            "encoder_depth": 5,
            "in_channels": 1,
            "out_channels": 1,
            "pyramid_channels": 256,
            "segmentation_channels": 128,
            "merge_policy": "add",
            "dropout": 0.2,
            "aux_heads_count": 4,
        },
        "teacher": {
            "ema_theta": 0.99,
            "freeze_during_mc": True,
            "inference_mode": True,
        },
        "noise": {
            "type": "Gaussian",
            "sigma": 0.01,
            "clamp_min": -0.1,
            "clamp_max": 0.1,
        },
        "uncertainty": {
            "metric": "voxel_entropy",
            "formula": "-2.0 * sum(p * log(p + 1e-6))",
            "dynamic_threshold": True,
            "threshold_start_factor": 0.75,
            "threshold_end_factor": 1.00,
            "threshold_rampup_steps": 4480,
        },
        "loss": {
            "deep_supervision": True,
            "aux_weights": [0.1, 0.2, 0.3, 0.4],
            "bce_weight": 0.5,
            "dice_weight": 0.5,
            "consistency_weight_max": 0.1,
            "consistency_rampup_epochs": 200,
        },
        "data": {
            "train_image_dir": "data/raw/DC1000_dataset/train/images",
            "train_label_dir": "data/raw/DC1000_dataset/train/labels",
            "val_image_cut_dir": "dataset/test/images_cut",
            "val_label_cut_dir": "dataset/test/labels_cut",
            "patch_size": 384,
            "batch_size": 8,
            "labeled_batch_size": 4,
            "unlabeled_batch_size": 4,
            "labeled_rates": {
                "0.1": 265,
                "0.2": 530,
                "0.5": 1325,
            },
            "active_rate": "0.1",
            "labeled_count": 265,
            "unlabeled_count": 2124,
            "total_training_patches": 2389,
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
            "scheduler": "LambdaLR",
            "poly_power": 0.9,
        },
        "validation": {
            "image_height": 768,
            "image_width": 1536,
            "patch_size": 384,
            "stride": 192,
            "num_patches": 21,
            "decision_threshold": 0.5,
            "val_interval": 1,
        },
        "output_paths": {
            "root_dir": f"outputs/experiments/{exp_id}",
            "checkpoints_dir": f"outputs/experiments/{exp_id}/checkpoints",
            "logs_dir": f"outputs/experiments/{exp_id}/logs",
            "evaluation_dir": f"outputs/experiments/{exp_id}/evaluation",
            "reports_dir": f"outputs/experiments/{exp_id}/reports",
            "history_csv": f"outputs/experiments/{exp_id}/training_history.csv",
        },
    }

    config_path = mc_configs_dir / f"{exp_id}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Generated {config_path}")

    # Create dedicated output directories
    out_dir = base_dir / "outputs" / "experiments" / exp_id
    for sub in ["checkpoints", "logs", "evaluation", "reports"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # Initialize empty training_history.csv with schema headers
    history_csv = out_dir / "training_history.csv"
    headers = [
        "epoch",
        "train_loss",
        "supervised_loss",
        "consistency_loss",
        "val_loss",
        "val_dice",
        "val_iou",
        "val_precision",
        "val_recall",
        "val_specificity",
        "val_f1",
        "learning_rate",
        "global_step",
        "epoch_duration",
        "cumulative_runtime",
        "max_foreground_prob",
        "pred_prevalence",
        "mean_uncertainty",
        "mean_entropy",
        "confidence_coverage",
    ]
    with open(history_csv, "w") as f:
        f.write(",".join(headers) + "\n")
    print(f"Initialized {history_csv}")

# Create outputs/experiments/MC_SAMPLING_COMPARISON.csv
comp_csv = base_dir / "outputs" / "experiments" / "MC_SAMPLING_COMPARISON.csv"
comp_headers = [
    "MC Samples",
    "Dice",
    "Sensitivity",
    "Precision",
    "IoU",
    "F1",
    "Specificity",
    "Accuracy",
    "Foreground Prevalence",
    "Mean Uncertainty",
    "Mean Entropy",
    "Confidence Coverage",
    "Training Time",
    "Inference Time",
    "Status",
]
rows = [
    ["T=5", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "NOT_RUN"],
    ["T=10", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "NOT_RUN"],
    ["T=20", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "NOT_RUN"],
    ["T=40", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "NOT_RUN"],
    ["T=80", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "NOT_RUN"],
    ["T=160", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "NOT_RUN"],
]
with open(comp_csv, "w") as f:
    f.write(",".join(comp_headers) + "\n")
    for row in rows:
        f.write(",".join(row) + "\n")
print(f"Generated {comp_csv}")

print("All 6 MC sampling configs, directories, and comparison schema created.")
