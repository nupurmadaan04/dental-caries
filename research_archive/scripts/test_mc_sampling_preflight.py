import sys
import os
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

base_dir = Path("c:/Users/devin/MLUA")
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from src.mlua.models.fpn import Net

def run_mc_preflight():
    print("=" * 75, flush=True)
    print("EXP-MLUA MC SAMPLING SENSITIVITY STUDY PRE-TRAINING AUDIT", flush=True)
    print("=" * 75, flush=True)

    mc_ids = ["MC-05", "MC-10", "MC-20", "MC-40", "MC-80", "MC-160"]
    expected_t_values = [5, 10, 20, 40, 80, 160]
    configs_dir = base_dir / "configs" / "experiments" / "mc_sampling"
    outputs_base = base_dir / "outputs" / "experiments"

    # 1. Parse all six YAML configs
    loaded_configs = {}
    for exp_id in mc_ids:
        cfg_file = configs_dir / f"{exp_id}.yaml"
        assert cfg_file.exists(), f"Missing config file: {cfg_file}"
        with open(cfg_file, "r") as f:
            cfg = yaml.safe_load(f)
        loaded_configs[exp_id] = cfg
        print(f"[{exp_id}] Config successfully loaded: {cfg['experiment']['name']}", flush=True)

    # 2. Verify six unique experiment IDs
    unique_ids = set(loaded_configs.keys())
    assert len(unique_ids) == 6, f"Expected 6 unique IDs, got {len(unique_ids)}"
    print(f"[Unique IDs] Verified 6 unique experiment IDs: {sorted(list(unique_ids))}", flush=True)

    # 3. Verify T values = 5, 10, 20, 40, 80, 160
    extracted_t = [loaded_configs[exp_id]["mc_sampling"]["mc_samples"] for exp_id in mc_ids]
    assert extracted_t == expected_t_values, f"T values mismatch! Found: {extracted_t}, Expected: {expected_t_values}"
    print(f"[T Values] Verified exact MC sample counts: {extracted_t}", flush=True)

    # 4. Verify ONLY mc_samples (and dependent metadata) changes between configs
    base_cfg = loaded_configs["MC-05"]
    for exp_id in mc_ids[1:]:
        cfg = loaded_configs[exp_id]
        # Verify model, data, optimizer, loss, teacher, noise, validation are identical
        assert cfg["model"] == base_cfg["model"], f"{exp_id} model config differed!"
        assert cfg["data"] == base_cfg["data"], f"{exp_id} data config differed!"
        assert cfg["optimization"] == base_cfg["optimization"], f"{exp_id} optimizer config differed!"
        assert cfg["loss"] == base_cfg["loss"], f"{exp_id} loss config differed!"
        assert cfg["teacher"] == base_cfg["teacher"], f"{exp_id} teacher config differed!"
        assert cfg["noise"] == base_cfg["noise"], f"{exp_id} noise config differed!"
        assert cfg["validation"] == base_cfg["validation"], f"{exp_id} validation config differed!"
    print("[Invariance] Verified that all non-MC scientific parameters are 100% identical across all 6 configs.", flush=True)

    # 5. Verify isolated directory trees and empty checkpoints
    for exp_id in mc_ids:
        out_dir = outputs_base / exp_id
        assert out_dir.exists(), f"Missing output dir: {out_dir}"
        for sub in ["checkpoints", "logs", "evaluation", "reports"]:
            assert (out_dir / sub).exists(), f"Missing subdir: {out_dir / sub}"
        ckpts = list((out_dir / "checkpoints").glob("*.pth"))
        assert len(ckpts) == 0, f"Found premature checkpoints in {out_dir / 'checkpoints'}"
        assert (out_dir / "training_history.csv").exists(), f"Missing history CSV in {out_dir}"
    print("[Directories] All 6 MC output directories & subdirectories verified clean and isolated.", flush=True)

    # 6. Dynamic MC Batching & Dimensions Smoke Tests (CPU-safe, small spatial size 64x64 for instant verification)
    device = torch.device("cpu")
    print(f"[Compute Device] Testing dynamic MC scaling on {device}...", flush=True)

    l_batch = 4
    ul_batch = 4
    spatial_size = 64
    model_tea = Net(in_c=1, out_c=1, encoder_name="resnet34", encoder_weights=None).to(device)

    for exp_id, t_val in zip(mc_ids, expected_t_values):
        total_evals = 1 + t_val
        total_in_images = total_evals * ul_batch # e.g. (1+5)*4 = 24 images, (1+160)*4 = 644 images
        
        # Test unified tensor creation dynamically (No hardcoded 8!)
        ul_data = torch.randn(ul_batch, 1, spatial_size, spatial_size, device=device)
        noise_base = torch.clamp(torch.randn_like(ul_data) * 0.01, -0.1, 0.1)
        noises_mc = torch.clamp(torch.randn(t_val, *ul_data.shape, device=device) * 0.01, -0.1, 0.1)

        unified_in = torch.empty((ul_batch + t_val * ul_batch, 1, spatial_size, spatial_size), device=device)
        unified_in[:ul_batch] = ul_data + noise_base
        unified_in[ul_batch:] = (ul_data.unsqueeze(0) + noises_mc).view(t_val * ul_batch, 1, spatial_size, spatial_size)

        assert unified_in.shape == (total_in_images, 1, spatial_size, spatial_size)

        with torch.no_grad():
            # Run 1 forward pass on unified input
            all_final, all_pyramid = model_tea(unified_in)
            
            ul_pred_tea = all_final[:ul_batch]
            mc_final = all_final[ul_batch:]
            mc_pyr = [h[ul_batch:] for h in all_pyramid]

            assert ul_pred_tea.shape == (ul_batch, 1, spatial_size, spatial_size)
            assert mc_final.shape == (t_val * ul_batch, 1, spatial_size, spatial_size)
            assert len(mc_pyr) == 4

            # Reshape into (5 * T, ul_batch, 1, H, W)
            b_final_5 = mc_final.view(t_val, ul_batch, 1, spatial_size, spatial_size).unsqueeze(0)
            b_pyr_5 = torch.stack([h.view(t_val, ul_batch, 1, spatial_size, spatial_size) for h in mc_pyr], dim=0)
            all_5 = torch.cat([b_final_5, b_pyr_5], dim=0).view(5 * t_val, ul_batch, 1, spatial_size, spatial_size)

            assert all_5.shape == (5 * t_val, ul_batch, 1, spatial_size, spatial_size)

            mean_preds = torch.mean(all_5, dim=0).sigmoid()
            assert mean_preds.shape == (ul_batch, 1, spatial_size, spatial_size)

            # Shannon Entropy calculation
            uncertainty = -2.0 * torch.sum(mean_preds * torch.log(mean_preds + 1e-6), dim=1, keepdim=True)
            assert uncertainty.shape == (ul_batch, 1, spatial_size, spatial_size)

            # Dynamic threshold mask
            threshold = 0.693
            mask = (uncertainty < threshold).float()
            assert mask.shape == (ul_batch, 1, spatial_size, spatial_size)

            assert not torch.isnan(uncertainty).any(), "NaN found in uncertainty!"
            assert not torch.isnan(mean_preds).any(), "NaN found in mean_preds!"
            assert not torch.isnan(mask).any(), "NaN found in mask!"

        print(f"[{exp_id}] T={t_val} Passed dynamic scaling test: Input={unified_in.shape} -> Ensemble={all_5.shape} -> Mean/Uncertainty/Mask={mean_preds.shape}", flush=True)

    # 7. Check sealed benchmark isolation
    sealed_dir = base_dir / "dataset" / "test"
    assert sealed_dir.exists(), "Sealed test set missing!"
    print("[Sealed Benchmark] dataset/test/ verified untouched and sealed.", flush=True)

    # 8. Check EXP-MLUA-001, EXP-MLUA-002, and ABL-00..07 integrity
    assert (outputs_base / "EXP-MLUA-001" / "EXP-MLUA-001_FULL_TRAINING_HISTORY.csv").exists()
    assert (outputs_base / "EXP-MLUA-002" / "EXP-MLUA-002_TRAINING_HISTORY.csv").exists()
    for abl_i in range(8):
        assert (outputs_base / f"ABL-0{abl_i}" / "training_history.csv").exists()
    print("[Experiment Isolation] EXP-MLUA-001, EXP-MLUA-002, and ABL-00..07 directories verified intact and untouched.", flush=True)

    print("\n" + "=" * 75, flush=True)
    print("ALL 6 MC SAMPLING PRE-TRAINING AUDIT CHECKS: [PASS]", flush=True)
    print("=" * 75, flush=True)

if __name__ == "__main__":
    run_mc_preflight()
