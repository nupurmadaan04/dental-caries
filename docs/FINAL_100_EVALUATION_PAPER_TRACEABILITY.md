# FINAL 100-IMAGE EVALUATION PAPER TRACEABILITY & PROTOCOL
**Project**: Standalone Dental Panoramic Caries Segmentation (MLUA)  
**Document**: `docs/FINAL_100_EVALUATION_PAPER_TRACEABILITY.md`  
**Classification**: Paper-Aligned 100-Image Evaluation Protocol on Dental Panoramic Radiographs  
**Status**: VERIFIED & DOCUMENTED

---

## 1. Evaluation Architecture Overview

The evaluation framework provides a standardized, sealed benchmark protocol to evaluate trained MLUA models on the 100 panoramic test cases of the DC1000 dataset without data leakage.

> *Statement of Distinction*:
> - **PAPER-REPORTED**: Published MLUA benchmark results on 3D cardiac CT/MRI slices (ACDC/LA datasets).
> - **OUR IMPLEMENTATION**: Adapted 2D dental panoramic evaluation pipeline processing 21 overlapping $384 \times 384$ radiograph patches reconstructed into $768 \times 1536$ panoramic evaluation space.

---

## 2. End-to-End Evaluation Protocol Mapping

```
                 [ 100 Panoramic Test Cases (768 x 1536) ]
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
[ 21 Patches per Case (384 x 384) ]                     [ Ground Truth Mask (768 x 1536) ]
(Stride: 192 x 192, 3x7 Grid)                                  │
           │                                                   │
[ Trained MLUA ResNet-34 FPN ]                                 │
           │                                                   │
[ 21 Patch Prediction Maps (384 x 384) ]                       │
           │                                                   │
[ Sliding-Window Overlap Averaging ]                           │
           │                                                   │
           ▼                                                   ▼
[ Reconstructed Panoramic Map (768 x 1536) ] ─────────► [ Binarization at tau = 0.50 ]
                                                               │
                                                               ▼
                                              [ Case-Level & Summary Metrics ]
                                              (Micro & Macro Dice, IoU, Rec, Prec, Spec)
```

| Evaluation Step | Paper Specification | Standalone DC1000 Implementation | Technical Rationale |
| :--- | :--- | :--- | :--- |
| **Test Case Count** | Full held-out test split | **100 Panoramic Cases** (`dataset/test/`) | Standardized DC1000 benchmark evaluation |
| **Input Representation** | 3D volumetric slices | **$768 \times 1536$ Panoramic Radiographs** | Standard dental panoramic aspect ratio |
| **Patch Extraction** | $256 \times 256$ / $112 \times 112 \times 80$ | **21 Patches ($384 \times 384$) at stride 192** | $3 \times 7$ grid covering full anatomical span |
| **Panoramic Reconstruction** | Slice stacking / volumetric stitching | **Arithmetic Overlap Averaging** | Exact pixel overlap normalization canvas |
| **Decision Threshold** | $\tau = 0.50$ | **$\tau = 0.50$ (Explicit Configuration)** | Fixed paper standard; no test set tuning |
| **Primary Metrics** | Dice, Sensitivity/Recall, Precision | **Dice, Sensitivity/Recall, Precision** | Primary diagnostic criteria for caries detection |
| **Diagnostic Metrics** | IoU, ASD, 95HD | **IoU, F1, Specificity, TP/FP/FN/TN Counts** | Full pixel confusion transparency |
| **Statistical Aggregation**| Case average | **Dual MACRO (Case Mean) & MICRO (Global)**| Unbiased case vs global pixel assessment |

---

## 3. Strict Sealing & Safety Safeguards

1. **Read-Only Test Policy**: `dataset/test/` is strictly read-only. The evaluation script refuses write operations inside the test directory.
2. **Zero Test Tuning**: The decision threshold ($\tau = 0.50$) is fixed by protocol. Threshold searching or post-processing tuning on the 100-case test set is strictly prohibited.
3. **Dedicated Output Tree**: All results are written exclusively to `outputs/evaluation/final_100/`.
4. **Reproducibility Manifest**: Complete case inventory and resolution parameters are recorded in `outputs/evaluation/final_100/TEST_SET_MANIFEST.json`.
