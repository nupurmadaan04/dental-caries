import os
import sys
import json
from pathlib import Path
import torch

def main():
    base_dir = Path.cwd()
    ckpt_path = base_dir / "outputs" / "experiments" / "EXP-MLUA-003" / "checkpoints" / "EXP-MLUA-003_LATEST.pth"
    
    if not ckpt_path.exists():
        print(f"Checkpoint not found at {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch", None)
    global_step = checkpoint.get("global_step", None)
    
    tea_state = checkpoint["model_tea_state_dict"]
    stu_state = checkpoint["model_stu_state_dict"]

    bn_keys = [k for k in tea_state if k.endswith("running_mean")]
    
    records = []
    default_mean_count = 0
    default_var_count = 0
    default_nbt_count = 0

    for mean_key in bn_keys:
        prefix = mean_key[:-len(".running_mean")]
        var_key = f"{prefix}.running_var"
        nbt_key = f"{prefix}.num_batches_tracked"

        t_mean = tea_state[mean_key].float()
        s_mean = stu_state[mean_key].float()
        t_var = tea_state[var_key].float()
        s_var = stu_state[var_key].float()
        t_nbt = tea_state[nbt_key].item()
        s_nbt = stu_state[nbt_key].item()

        t_mean_is_default = bool(torch.all(t_mean == 0.0).item())
        t_var_is_default = bool(torch.all(t_var == 1.0).item())
        t_nbt_is_default = (t_nbt == 0)

        if t_mean_is_default: default_mean_count += 1
        if t_var_is_default: default_var_count += 1
        if t_nbt_is_default: default_nbt_count += 1

        records.append({
            "layer": prefix,
            "tea_mean_range": [float(t_mean.min().item()), float(t_mean.max().item())],
            "stu_mean_range": [float(s_mean.min().item()), float(s_mean.max().item())],
            "tea_var_range": [float(t_var.min().item()), float(t_var.max().item())],
            "stu_var_range": [float(s_var.min().item()), float(s_var.max().item())],
            "tea_nbt": int(t_nbt),
            "stu_nbt": int(s_nbt),
            "mean_abs_diff": float(torch.abs(t_mean - s_mean).max().item()),
            "var_abs_diff": float(torch.abs(t_var - s_var).max().item())
        })

    summary = {
        "epoch": epoch,
        "global_step": global_step,
        "total_bn_layers": len(bn_keys),
        "default_mean_count": default_mean_count,
        "default_var_count": default_var_count,
        "default_nbt_count": default_nbt_count,
        "records": records
    }

    out_file = base_dir / "outputs" / "experiments" / "EXP-MLUA-003" / "diagnostics" / "checkpoint_buffer_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Checkpoint Epoch: {epoch}, Global Step: {global_step}")
    print(f"Total BatchNorm Layers: {len(bn_keys)}")
    print(f"Teacher Default Means: {default_mean_count}/{len(bn_keys)}")
    print(f"Teacher Default Vars:  {default_var_count}/{len(bn_keys)}")
    print(f"Teacher Default NBT:   {default_nbt_count}/{len(bn_keys)}")
    print(f"Sample layer4.2.bn1 Teacher running_var max: {tea_state['encoder.layer4.2.bn1.running_var'].max().item():.4f}")
    print(f"Sample layer4.2.bn1 Student running_var max: {stu_state['encoder.layer4.2.bn1.running_var'].max().item():.4f}")

if __name__ == "__main__":
    main()
