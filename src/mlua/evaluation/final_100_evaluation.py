"""
MLUA Final 100-Image Panoramic Evaluation Pipeline
Implements 21-patch sliding-window panoramic inference (384x384 at stride 192),
reconstruction to 768x1536 panoramic space, and dual Micro/Macro metric evaluation.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

base_dir = Path("c:/Users/devin/MLUA")
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from src.mlua.models.fpn import Net


def extract_panoramic_patches(
    image: np.ndarray,
    patch_size: int = 384,
    stride: int = 192,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Extracts 21 overlapping 384x384 patches from a 768x1536 image at stride 192.
    """
    h, w = image.shape[:2]
    assert (h, w) == (768, 1536), f"Expected panoramic dimensions (768, 1536), got ({h}, {w})"

    patches = []
    coords = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y : y + patch_size, x : x + patch_size]
            patches.append(patch)
            coords.append((y, x))

    assert len(patches) == 21, f"Expected exactly 21 patches, got {len(patches)}"
    return patches, coords


def reconstruct_panoramic(
    patch_preds: List[np.ndarray],
    coords: List[Tuple[int, int]],
    full_shape: Tuple[int, int] = (768, 1536),
    patch_size: int = 384,
) -> np.ndarray:
    """
    Reconstructs full panoramic probability map by averaging overlapping 384x384 patch predictions.
    """
    canvas = np.zeros(full_shape, dtype=np.float32)
    count_map = np.zeros(full_shape, dtype=np.float32)

    for patch, (y, x) in zip(patch_preds, coords):
        canvas[y : y + patch_size, x : x + patch_size] += patch
        count_map[y : y + patch_size, x : x + patch_size] += 1.0

    assert np.all(count_map >= 1.0), "Found unvisited pixels during panoramic reconstruction!"
    reconstructed = canvas / count_map
    return reconstructed


def calculate_case_metrics(
    pred_prob: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.50,
    case_id: str = "unknown",
) -> Dict[str, Any]:
    """
    Computes comprehensive binary segmentation metrics for a single panoramic case.
    """
    pred_bin = (pred_prob >= threshold).astype(np.uint8)
    gt_bin = (gt_mask > 0.5).astype(np.uint8)

    tp = int(np.logical_and(pred_bin == 1, gt_bin == 1).sum())
    fp = int(np.logical_and(pred_bin == 1, gt_bin == 0).sum())
    fn = int(np.logical_and(pred_bin == 0, gt_bin == 1).sum())
    tn = int(np.logical_and(pred_bin == 0, gt_bin == 0).sum())

    eps = 1e-6
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    spec = tn / (tn + fp + eps)
    f1 = dice

    gt_fg_pixels = int(gt_bin.sum())
    pred_fg_pixels = int(pred_bin.sum())
    total_pixels = int(gt_bin.size)

    pred_prevalence = pred_fg_pixels / total_pixels
    gt_prevalence = gt_fg_pixels / total_pixels

    return {
        "case_id": case_id,
        "dice": round(float(dice), 5),
        "iou": round(float(iou), 5),
        "precision": round(float(prec), 5),
        "recall": round(float(rec), 5),
        "specificity": round(float(spec), 5),
        "f1": round(float(f1), 5),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "gt_foreground_pixels": gt_fg_pixels,
        "pred_foreground_pixels": pred_fg_pixels,
        "pred_prevalence": round(float(pred_prevalence), 6),
        "gt_prevalence": round(float(gt_prevalence), 6),
        "threshold": threshold,
    }


class Final100Evaluator:
    """
    Final 100-Image Benchmark Evaluation Manager.
    Enforces strict read-only isolation on dataset/test/ and outputs standardized evaluation tables.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        self.cfg = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = float(self.cfg["evaluation"]["threshold"])
        self.out_dir = base_dir / self.cfg["evaluation"]["output_dir"]
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self, checkpoint_path: Path) -> nn.Module:
        """Loads and prepares trained model for evaluation."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model = Net(in_c=1, out_c=1, encoder_name="resnet34", encoder_weights=None)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load weights from either student or teacher state dict
        if "model_stu_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_stu_state_dict"])
        elif "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif "model_tea_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_tea_state_dict"])
        else:
            model.load_state_dict(ckpt)

        model.to(self.device)
        model.eval()
        return model

    def evaluate_panoramic_case(
        self,
        model: nn.Module,
        panoramic_img: np.ndarray,
        gt_mask: np.ndarray,
        case_id: str,
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Runs full 21-patch sliding-window inference and panoramic reconstruction on a single case.
        """
        patches, coords = extract_panoramic_patches(
            panoramic_img,
            patch_size=self.cfg["evaluation"]["patch_size"],
            stride=self.cfg["evaluation"]["stride"],
        )

        # Batch 21 patches into tensor
        patch_tensors = []
        for p in patches:
            p_norm = p.astype(np.float32) / 255.0 if p.max() > 1.0 else p.astype(np.float32)
            if p_norm.ndim == 2:
                p_norm = np.expand_dims(p_norm, 0)
            elif p_norm.ndim == 3 and p_norm.shape[2] == 3:
                p_norm = np.mean(p_norm, axis=2, keepdims=True).transpose(2, 0, 1)
            patch_tensors.append(torch.from_numpy(p_norm))

        batch_tensor = torch.stack(patch_tensors, dim=0).to(self.device) # [21, 1, 384, 384]

        with torch.inference_mode():
            pred_fused, _ = model(batch_tensor)
            pred_probs = torch.sigmoid(pred_fused).squeeze(1).cpu().numpy() # [21, 384, 384]

        reconstructed_prob = reconstruct_panoramic(
            list(pred_probs),
            coords,
            full_shape=(768, 1536),
            patch_size=self.cfg["evaluation"]["patch_size"],
        )

        metrics = calculate_case_metrics(
            reconstructed_prob,
            gt_mask,
            threshold=self.threshold,
            case_id=case_id,
        )
        return metrics, reconstructed_prob

    def compute_summary(self, case_records: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Computes both MACRO and MICRO aggregate statistics across all cases.
        """
        if not case_records:
            return pd.DataFrame()

        df_cases = pd.DataFrame(case_records)

        # 1. Macro Aggregation (Arithmetic mean across cases)
        macro_dice = float(df_cases["dice"].mean())
        macro_iou = float(df_cases["iou"].mean())
        macro_prec = float(df_cases["precision"].mean())
        macro_rec = float(df_cases["recall"].mean())
        macro_spec = float(df_cases["specificity"].mean())
        macro_f1 = float(df_cases["f1"].mean())

        # 2. Micro Aggregation (Global pixel confusion sum)
        total_tp = int(df_cases["tp"].sum())
        total_fp = int(df_cases["fp"].sum())
        total_fn = int(df_cases["fn"].sum())
        total_tn = int(df_cases["tn"].sum())
        eps = 1e-6

        micro_dice = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn + eps)
        micro_iou = total_tp / (total_tp + total_fp + total_fn + eps)
        micro_prec = total_tp / (total_tp + total_fp + eps)
        micro_rec = total_tp / (total_tp + total_fn + eps)
        micro_spec = total_tn / (total_tn + total_fp + eps)
        micro_f1 = micro_dice

        summary_rows = [
            {
                "aggregation_type": "MACRO (Case Mean)",
                "dice": round(macro_dice, 5),
                "iou": round(macro_iou, 5),
                "precision": round(macro_prec, 5),
                "recall": round(macro_rec, 5),
                "specificity": round(macro_spec, 5),
                "f1": round(macro_f1, 5),
                "total_cases": len(df_cases),
                "threshold": self.threshold,
            },
            {
                "aggregation_type": "MICRO (Global Pixel Sum)",
                "dice": round(float(micro_dice), 5),
                "iou": round(float(micro_iou), 5),
                "precision": round(float(micro_prec), 5),
                "recall": round(float(micro_rec), 5),
                "specificity": round(float(micro_spec), 5),
                "f1": round(float(micro_f1), 5),
                "total_cases": len(df_cases),
                "threshold": self.threshold,
            }
        ]
        return pd.DataFrame(summary_rows)
