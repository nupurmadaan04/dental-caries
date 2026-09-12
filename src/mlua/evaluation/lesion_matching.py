"""
MLUA Lesion Extraction & Component Matching Utility
Provides connected-component lesion extraction, size grouping, and 1-to-1 bipartite/greedy IoU matching.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.ndimage import label, center_of_mass, find_objects


def extract_connected_lesions(
    binary_mask: np.ndarray,
    case_id: str = "unknown",
    connectivity: int = 8,
) -> List[Dict[str, Any]]:
    """
    Extracts individual connected foreground components from a 2D binary mask.
    
    Size categories:
    - SMALL:  area < 300 pixels
    - MEDIUM: 300 <= area <= 1000 pixels
    - LARGE:  area > 1000 pixels
    """
    mask_bin = (binary_mask > 0.5).astype(np.uint8)
    structure = np.ones((3, 3), dtype=np.uint8) if connectivity == 8 else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labeled_array, num_features = label(mask_bin, structure=structure)

    lesions = []
    if num_features == 0:
        return lesions

    slices = find_objects(labeled_array)
    centroids = center_of_mass(mask_bin, labeled_array, range(1, num_features + 1))
    if num_features == 1:
        centroids = [centroids]

    for idx in range(1, num_features + 1):
        comp_mask = (labeled_array == idx)
        area = int(np.sum(comp_mask))
        if area == 0:
            continue

        if area < 300:
            size_group = "SMALL"
        elif 300 <= area <= 1000:
            size_group = "MEDIUM"
        else:
            size_group = "LARGE"

        sl = slices[idx - 1]
        r_min, r_max = sl[0].start, sl[0].stop
        c_min, c_max = sl[1].start, sl[1].stop
        height = r_max - r_min
        width = c_max - c_min

        centroid = centroids[idx - 1]
        c_row, c_col = float(centroid[0]), float(centroid[1])

        lesions.append({
            "case_id": case_id,
            "lesion_id": idx,
            "area_pixels": area,
            "size_group": size_group,
            "bounding_box": (r_min, c_min, r_max, c_max),
            "centroid": (c_row, c_col),
            "width": width,
            "height": height,
            "mask": comp_mask,
        })

    return lesions


def match_lesions(
    gt_lesions: List[Dict[str, Any]],
    pred_lesions: List[Dict[str, Any]],
    iou_threshold: float = 0.10,
) -> List[Dict[str, Any]]:
    """
    Performs greedy 1-to-1 matching between Ground-Truth and Predicted lesion components.
    
    Matching Rule:
    - Calculates IoU for all (GT_i, Pred_j) candidate pairs.
    - Greedily assigns highest IoU pair first if IoU >= iou_threshold.
    - Each GT lesion is matched to at most one predicted component.
    - Each predicted component is matched to at most one GT lesion.
    """
    results = []
    if not gt_lesions:
        return results

    if not pred_lesions:
        for gt in gt_lesions:
            results.append({
                "case_id": gt["case_id"],
                "lesion_id": gt["lesion_id"],
                "size_group": gt["size_group"],
                "gt_area_pixels": gt["area_pixels"],
                "pred_area_pixels": 0,
                "intersection_pixels": 0,
                "union_pixels": gt["area_pixels"],
                "lesion_iou": 0.0,
                "matched": False,
                "matched_pred_id": None,
            })
        return results

    # Compute IoU matrix
    iou_matrix = np.zeros((len(gt_lesions), len(pred_lesions)), dtype=np.float32)
    inter_matrix = np.zeros((len(gt_lesions), len(pred_lesions)), dtype=np.int64)
    union_matrix = np.zeros((len(gt_lesions), len(pred_lesions)), dtype=np.int64)

    for i, gt in enumerate(gt_lesions):
        for j, pred in enumerate(pred_lesions):
            inter = int(np.logical_and(gt["mask"], pred["mask"]).sum())
            union = int(np.logical_or(gt["mask"], pred["mask"]).sum())
            inter_matrix[i, j] = inter
            union_matrix[i, j] = union
            iou_matrix[i, j] = inter / max(union, 1)

    matched_gt = set()
    matched_pred = set()
    matches: Dict[int, Tuple[int, float, int, int]] = {}

    # Sort candidate pairs by IoU descending
    pairs = []
    for i in range(len(gt_lesions)):
        for j in range(len(pred_lesions)):
            if iou_matrix[i, j] >= iou_threshold:
                pairs.append((iou_matrix[i, j], i, j))
    pairs.sort(key=lambda x: x[0], reverse=True)

    for iou_val, i, j in pairs:
        if i not in matched_gt and j not in matched_pred:
            matched_gt.add(i)
            matched_pred.add(j)
            matches[i] = (j, float(iou_val), int(inter_matrix[i, j]), int(union_matrix[i, j]))

    for i, gt in enumerate(gt_lesions):
        if i in matches:
            j, iou_val, inter, union = matches[i]
            pred_item = pred_lesions[j]
            results.append({
                "case_id": gt["case_id"],
                "lesion_id": gt["lesion_id"],
                "size_group": gt["size_group"],
                "gt_area_pixels": gt["area_pixels"],
                "pred_area_pixels": pred_item["area_pixels"],
                "intersection_pixels": inter,
                "union_pixels": union,
                "lesion_iou": round(iou_val, 4),
                "matched": True,
                "matched_pred_id": pred_item["lesion_id"],
            })
        else:
            results.append({
                "case_id": gt["case_id"],
                "lesion_id": gt["lesion_id"],
                "size_group": gt["size_group"],
                "gt_area_pixels": gt["area_pixels"],
                "pred_area_pixels": 0,
                "intersection_pixels": 0,
                "union_pixels": gt["area_pixels"],
                "lesion_iou": 0.0,
                "matched": False,
                "matched_pred_id": None,
            })

    return results
