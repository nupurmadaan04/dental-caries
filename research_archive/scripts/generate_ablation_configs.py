import os
import yaml
from pathlib import Path

base_dir = Path("c:/Users/devin/MLUA")
ablation_configs_dir = base_dir / "configs" / "experiments" / "ablation"
ablation_configs_dir.mkdir(parents=True, exist_ok=True)

ablations = [
    {
        "id": "ABL-00",
        "name": "Baseline",
        "desc": "Baseline SSL without MLUA disturbances (No EMA, No Gaussian/MC, No Deep Supervision)",
        "iterative": False,
        "noisy": False,
        "multiscale": False,
    },
    {
        "id": "ABL-01",
        "name": "Iterative Only",
        "desc": "Iterative Disturbance Only (EMA Teacher ON, Noisy OFF, Deep Supervision OFF)",
        "iterative": True,
        "noisy": False,
        "multiscale": False,
    },
    {
        "id": "ABL-02",
        "name": "Noisy Only",
        "desc": "Noisy Disturbance Only (Gaussian Noise + MC T=8 ON, EMA OFF, Deep Supervision OFF)",
        "iterative": False,
        "noisy": True,
        "multiscale": False,
    },
    {
        "id": "ABL-03",
        "name": "Multi-scale Only",
        "desc": "Multi-scale Disturbance Only (Deep Supervision ON, EMA OFF, Noisy OFF)",
        "iterative": False,
        "noisy": False,
        "multiscale": True,
    },
    {
        "id": "ABL-04",
        "name": "Iterative + Noisy",
        "desc": "Iterative & Noisy Disturbance (EMA Teacher ON + MC Noise ON, Deep Supervision OFF)",
        "iterative": True,
        "noisy": True,
        "multiscale": False,
    },
    {
        "id": "ABL-05",
        "name": "Iterative + Multi-scale",
        "desc": "Iterative & Multi-scale Disturbance (EMA Teacher ON + Deep Supervision ON, Noisy OFF)",
        "iterative": True,
        "noisy": False,
        "multiscale": True,
    },
    {
        "id": "ABL-06",
        "name": "Noisy + Multi-scale",
        "desc": "Noisy & Multi-scale Disturbance (MC Noise ON + Deep Supervision ON, EMA OFF)",
        "iterative": False,
        "noisy": True,
        "multiscale": True,
    },
    {
        "id": "ABL-07",
        "name": "Full MLUA",
        "desc": "Full MLUA Framework (Iterative ON + Noisy ON + Multi-scale ON)",
        "iterative": True,
        "noisy": True,
        "multiscale": True,
    },
]

for abl in ablations:
    exp_id = abl["id"]
    cfg = {
        "experiment": {
            "experiment_id": exp_id,
            "name": f"{exp_id}_{abl['name'].replace(' ', '_')}",
            "description": abl["desc"],
            "seed": 42,
            "ssl_enabled": True,
            "max_epochs": 200,
            "precision": 16,
        },
        "disturbances": {
            "iterative_disturbance": abl["iterative"],
            "noisy_disturbance": abl["noisy"],
            "multiscale_disturbance": abl["multiscale"],
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
        "ssl": {
            "ema_enabled": abl["iterative"],
            "ema_theta": 0.99 if abl["iterative"] else 0.0,
            "mc_enabled": abl["noisy"],
            "mc_iterations": 8 if abl["noisy"] else 1,
            "noise_enabled": abl["noisy"],
            "noise_sigma": 0.01 if abl["noisy"] else 0.0,
            "noise_clamp": 0.1 if abl["noisy"] else 0.0,
            "dynamic_threshold": abl["noisy"],
            "threshold_rampup_steps": 4480,
            "threshold_start_factor": 0.75,
            "threshold_end_factor": 1.00,
            "consistency_weight_max": 0.1,
            "consistency_rampup_epochs": 200,
        },
        "loss": {
            "deep_supervision": abl["multiscale"],
            "aux_weights": [0.1, 0.2, 0.3, 0.4] if abl["multiscale"] else [0.0, 0.0, 0.0, 0.0],
            "bce_weight": 0.5,
            "dice_weight": 0.5,
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

    config_path = ablation_configs_dir / f"{exp_id}.yaml"
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
    ]
    with open(history_csv, "w") as f:
        f.write(",".join(headers) + "\n")
    print(f"Initialized {history_csv}")

print("All 8 ablation configs & directory structures successfully created.")
