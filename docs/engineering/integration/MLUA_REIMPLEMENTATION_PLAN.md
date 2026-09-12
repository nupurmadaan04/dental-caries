# MLUA Standalone Reimplementation Plan & System Blueprint

## 1. Architectural Principles & Reimplementation Strategy

When constructing the new standalone MLUA project, we must enforce a strict separation between research core logic and production engineering.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        STANDALONE MLUA PLATFORM                        │
├───────────────────────────────────┬────────────────────────────────────┤
│           KEEP EXACTLY            │          ADAPT CAREFULLY           │
│         (Scientific Core)         │       (Engineering & UX)           │
├───────────────────────────────────┼────────────────────────────────────┤
│ • ResNet-34 FPN Architecture      │ • Replace hard-coded paths with    │
│ • 4 Auxiliary Heads + 1 Fused     │   cross-platform pathlib & configs │
│ • Monte Carlo Perturbation (T=8)  │ • FastAPI Inference REST Backend   │
│ • 40 multi-level predictions / smp│ • Real-time Uncertainty Heatmaps   │
│ • Dynamic Ramp-up Threshold       │ • Modern Dark-Mode Dental Web UI   │
│ • Deep Supervision Loss Formula   │ • PyTorch Lightning 2.x Modular    │
│ • EMA Alpha Ramp-up Equation      │ • Robust Multi-GPU / CPU support   │
└───────────────────────────────────┴────────────────────────────────────┘
```

---

## 2. Standalone Codebase Architecture Blueprint

```text
mlua-standalone/
│
├── configs/
│   ├── default.yaml                 # Base hyperparameters (LR, epochs, batch)
│   ├── ssl_mlua_10.yaml             # 10% labeled partition config
│   ├── ssl_mlua_20.yaml             # 20% labeled partition config
│   └── inference.yaml               # Inference thresholds and sliding window
│
├── src/
│   ├── mlua/
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── fpn.py               # Clean Net with ResNet-34 + FPN + 4 Aux Heads
│   │   │   ├── decoder.py           # Top-down pyramid decoder and SegBlocks
│   │   │   └── heads.py             # Fused and auxiliary segmentation heads
│   │   │
│   │   ├── uncertainty/
│   │   │   ├── __init__.py
│   │   │   ├── monte_carlo.py       # T=8 MC multi-level perturbation sampler
│   │   │   └── entropy.py           # Voxel-wise entropy & dynamic thresholding
│   │   │
│   │   ├── losses/
│   │   │   ├── __init__.py
│   │   │   ├── dice.py              # BinaryDiceLoss
│   │   │   ├── supervision.py       # Deep supervision multi-head loss
│   │   │   └── consistency.py       # Masked sigmoid MSE consistency loss
│   │   │
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── dataset.py           # TrainDataset & ValDataset with pathlib
│   │   │   ├── sampler.py           # TwoStreamBatchSampler
│   │   │   └── transforms.py        # Synchronized spatial/photometric augs
│   │   │
│   │   ├── engine/
│   │   │   ├── __init__.py
│   │   │   ├── lightning_module.py  # PyTorch Lightning 2.x MLUAModule
│   │   │   └── trainer.py           # CLI trainer with TensorBoard & rich logging
│   │   │
│   │   └── evaluation/
│   │       ├── __init__.py
│   │       ├── sliding_window.py    # 50% overlap tiling & recomposition
│   │       └── metrics.py           # Dice, IoU, Sens, Spec, Prec calculation
│   │
│   ├── backend/
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes.py            # /predict, /uncertainty, /health
│   │   │   └── schemas.py           # Pydantic request/response models
│   │   ├── services/
│   │   │   ├── inference_engine.py  # Model loading and tile recomposition
│   │   │   └── heatmap_generator.py # Overlay and colormap generation
│   │   └── main.py                  # FastAPI application entrypoint
│   │
│   └── frontend/                    # Modern Vite + React or Vanilla Web UI
│       ├── index.html
│       ├── src/
│       └── assets/
│
├── checkpoints/                     # Model weights storage
├── docs/                            # Comprehensive documentation
├── tests/                           # Unit & integration tests
├── pyproject.toml
└── README.md
```

---

## 3. Frontend Requirements & UX Blueprint

The eventual web application will feature a modern, dark-mode dental visualization interface matching our established aesthetic:

1. **Panoramic Radiograph Upload & Interactive Viewer**:
   - Drag-and-drop DICOM / PNG / JPEG upload.
   - Zoomable viewport with high dynamic range brightness and contrast adjustment.
2. **Segmentation Overlay & Layer Toggling**:
   - Primary Fused Prediction contour and mask fill.
   - Multi-Scale Auxiliary Predictions (Level 2, Level 3, Level 4, Level 5 toggles).
3. **Uncertainty & Decision-Support Visualizations**:
   - Interactive Color Heatmap of voxel-wise Monte Carlo entropy.
   - Confidence threshold slider showing how the dynamic threshold filters uncertain boundary pixels.
   - Side-by-side comparison of Teacher vs Student prediction maps.
4. **Clinical Decision Metrics & Region Summary**:
   - Total detected caries lesion count.
   - Lesion bounding boxes, surface area, and tooth quadrant localization.
   - Model inference metadata (backbone, resolution, processing time).
5. **Research Transparency & Limitation Banner**:
   - Clear classification as an AI-assisted research prototype.

---

## 4. Medical & Research Safety Classification

> [!IMPORTANT]
> **Safety & Regulatory Classification**:
> The system is strictly categorized as a **Research AI-Assisted Dental Caries Segmentation Prototype and Decision-Support Demonstration**.
> 
> It is **NOT**:
> - An autonomous diagnostic medical device.
> - A clinically certified caries detection or staging system.
> - A replacement for licensed dental practitioner interpretation.
>
> All system outputs must include clear disclaimers stating that segmentation masks and uncertainty maps are provided solely for research evaluation and educational review.
