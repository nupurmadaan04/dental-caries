# EXP-MLUA-003 Integration Preflight Report

**Project:** EXP-MLUA-003 (ResNet-34 + FPN MLUA Semi-Supervised Segmentation)  
**Preflight Type:** Read-Only Engineering Integration Sanity Check  
**Date:** September 12, 2026  
**Status:** **PASS**  

---

## 1. Model

- **Experiment:** `EXP-MLUA-003`
- **Checkpoint:** [`outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E56_FINAL.pth`](file:///c:/Users/devin/MLUA/outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E56_FINAL.pth)
- **Epoch:** `56`
- **Global Step:** `7392`
- **Frozen Status:** **STRICTLY FROZEN** (weights, parameters, buffers, and architecture configuration).

---

## 2. Research Configuration

- **Preprocessing:** Grayscale conversion (`.convert("L")`), standard channel normalization (`/ 255.0` to $[0.0, 1.0]$ float32), preserved aspect ratio resizing to standard panoramic dimensions ($768 \times 1536$).
- **Patch Size & Extraction:** $384 \times 384$ pixels, 21 overlapping patches with horizontal and vertical stride of $192$ pixels ($3 \times 7$ spatial grid).
- **Reconstruction:** Overlap-averaged reconstruction accumulating probabilities across intersecting patch coordinates and normalizing by spatial patch overlap counts.
- **Operating Threshold:** $\tau = 0.50$ (selected via validation threshold sensitivity analysis).
- **Inference Mode:** `torch.inference_mode()` with frozen weights and evaluation mode enabled (`model.eval()`).

---

## 3. Checkpoint Verification

- **Loaded Successfully:** Yes (`torch.load` on CPU / GPU compatible).
- **Architecture Compatibility:** `Net(in_c=1, out_c=1, encoder_name="resnet34", encoder_weights=None)` perfectly instantiated.
- **Strict Key Matching:** `<All keys matched successfully>` (0 missing keys, 0 unexpected keys).
- **Finite Parameters:** 100% finite across all floating-point tensors:
  - NaN parameter count: `0`
  - Inf parameter count: `0`

---

## 4. Representative Inference

- **Representative Image Source:** `data/raw/DC1000_dataset/org_train_dataset/images/101.png` (Training set radiograph; strictly separate from sealed test set).
- **Raw Input Shape:** $1435 \times 2943$ pixels (Grayscale).
- **Standard Panoramic Shape:** $768 \times 1536$ pixels ($H \times W$).
- **Patch Count:** Exactly 21 overlapping patches extracted and batched as `[21, 1, 384, 384]`.
- **Raw Fused Output Logits Shape:** `[21, 1, 384, 384]` (NaN: `0`, Inf: `0`).
- **Reconstructed Output Shape:** $768 \times 1536$ continuous probability map.
- **Probability Statistics:**
  - Probability Min: `0.000000` ($1.233 \times 10^{-18}$)
  - Probability Max: `0.998185`
  - Probability Mean: `0.015858`
  - NaN Count: `0`
  - Inf Count: `0`
  - Range Validity: All values strictly within $[0.0, 1.0]$.

---

## 5. Binary Mask

- **Decision Threshold:** $\tau = 0.50$
- **Positive Pixel Count:** `7,271` pixels
- **Positive Pixel Percentage:** `0.6164%` of panoramic canvas ($7,271 / 1,179,648$)
- **Zero-Prediction Status:** `False` (active segmentation produced).
- *Engineering Note:* Non-empty mask confirms end-to-end signal propagation and thresholding integrity; it does not constitute clinical validation.

---

## 6. Reconstruction

- **Reconstructed Dimensions:** $768 \times 1536$ pixels.
- **Spatial Consistency:** Verified across all 21 patch coordinates $(y, x) \in \{0, 192, 384\} \times \{0, 192, 384, 576, 768, 960, 1152\}$.
- **Padding / Cropping Status:** Clean boundary alignment; no residual padding or spatial transposition detected.

---

## 7. Visualization Artifacts

The following temporary preflight visualization artifacts have been generated in [`outputs/diagnostics/FINAL_INTEGRATION_PREFLIGHT/`](file:///c:/Users/devin/MLUA/outputs/diagnostics/FINAL_INTEGRATION_PREFLIGHT/):

1. **Original Radiograph:** [`representative_original.png`](file:///c:/Users/devin/MLUA/outputs/diagnostics/FINAL_INTEGRATION_PREFLIGHT/representative_original.png) ($768 \times 1536$, Grayscale)
2. **Predicted Probability Map:** [`representative_probability.png`](file:///c:/Users/devin/MLUA/outputs/diagnostics/FINAL_INTEGRATION_PREFLIGHT/representative_probability.png) ($768 \times 1536$, Normalized Grayscale $[0, 255]$)
3. **Binary Segmentation Mask:** [`representative_mask.png`](file:///c:/Users/devin/MLUA/outputs/diagnostics/FINAL_INTEGRATION_PREFLIGHT/representative_mask.png) ($768 \times 1536$, $\{0, 255\}$ Binary)
4. **Composite Overlay:** [`representative_overlay.png`](file:///c:/Users/devin/MLUA/outputs/diagnostics/FINAL_INTEGRATION_PREFLIGHT/representative_overlay.png) ($768 \times 1536$, RGB with 60% Alpha Red Lesion Mask Overlay)
5. **Execution JSON:** [`preflight_metrics.json`](file:///c:/Users/devin/MLUA/outputs/diagnostics/FINAL_INTEGRATION_PREFLIGHT/preflight_metrics.json)

---

## 8. Sealed Test Protection

> **The sealed 100-case test set was not used during this integration preflight.**  
> The preflight was executed exclusively on a non-sealed training radiograph (`101.png`), ensuring 100% preservation and zero leakage for test benchmarks.

---

## 9. Result

# **PASS**
