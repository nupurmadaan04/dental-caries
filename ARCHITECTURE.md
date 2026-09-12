# MLUA System Architecture

> For comprehensive mathematical formulations and loss derivations, see [**`docs/ARCHITECTURE.md`**](docs/ARCHITECTURE.md).

## 1. High-Level Architecture

The **Multi-Level Uncertainty-Aware (MLUA)** system is an end-to-end framework for pixel-level dental caries segmentation and clinical radiology review from panoramic X-rays (OPGs).

```
+-----------------------------------------------------------------------------------+
|                            Input Panoramic Radiograph                             |
|                              (768 x 1536 Grayscale)                               |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        21 Overlapping Sub-Patches                                 |
|                       (384 x 384, Stride = 192 px)                                |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                             ResNet-34 Feature Encoder                             |
|                         Stages: C1 -> C2 -> C3 -> C4 -> C5                        |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                         Feature Pyramid Network (FPN)                             |
|                    Lateral Connections & Top-Down Merging                         |
|                         Pyramids: P2, P3, P4, P5 (256 ch)                         |
+-----------------------------------------------------------------------------------+
            |                    |                    |                    |
            v                    v                    v                    v
      +-----------+        +-----------+        +-----------+        +-----------+
      | Aux Head 1|        | Aux Head 2|        | Aux Head 3|        | Aux Head 4|
      | (1/8 Res) |        | (1/4 Res) |        | (1/2 Res) |        | (1/1 Res) |
      | α_1 = 0.1 |        | α_2 = 0.2 |        | α_3 = 0.3 |        | α_4 = 0.4 |
      +-----------+        +-----------+        +-----------+        +-----------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                   Monte Carlo Epistemic Uncertainty Gating                        |
|                     T = 8 Stochastic Passes with Dropout                          |
|             Mean μ(x) & Uncertainty Variance σ²(x) Pseudo-label Gating            |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                       2D Gaussian Patch Blending                                  |
|               Reconstructed 768 x 1536 Full Panoramic Mask                        |
|                       Operating Threshold τ = 0.50                                |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Pillars

1. **Multi-Scale Deep Supervision:** 4 auxiliary prediction heads branching from the FPN decoder stages ($1/8, 1/4, 1/2, \text{and } 1/1$) prevent vanishing gradients and reinforce fine enamel boundary detection.
2. **Epistemic Uncertainty Quantification ($T=8$):** Monte Carlo dropout forward passes calculate pixel-level variance $\sigma^2(x)$ to gate unconfident pseudo-labels during semi-supervised training.
3. **Synchronized Teacher EMA ($\theta = 0.99$):** Both weights and floating-point BatchNorm running statistics are synchronized continuously, ensuring numerical stability across 60 epochs (7,920 steps).
4. **Interactive Fullstack Clinical Interface:** React 18 + Vite frontend with multi-layer canvas overlay viewer, 4-tier clinical severity staging (Green/Yellow/Orange/Red), and vector-quality 2-page PDF radiology report generation.
