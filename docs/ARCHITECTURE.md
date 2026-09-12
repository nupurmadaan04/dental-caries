# MLUA Architecture & Theoretical Formulation

## 1. System Overview

The **Multi-Level Uncertainty-Aware (MLUA)** deep learning framework is an advanced semi-supervised neural network architecture engineered specifically for pixel-level dental caries segmentation in panoramic dental radiographs (Orthopantomograms / OPGs).

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

## 2. Multi-Scale Deep Supervision

To ensure strong gradient flow back to early convolutional layers and prevent vanishing gradients when segmenting subtle, low-contrast enamel lesions, the model deploys **4 multi-scale auxiliary heads** at different spatial resolutions ($1/8, 1/4, 1/2, \text{and } 1/1$):

$$\mathcal{L}_{\text{sup}} = \sum_{k=1}^{4} \alpha_k \left[ \mathcal{L}_{\text{BCE}}(P_k, Y) + \mathcal{L}_{\text{SoftDice}}(P_k, Y) \right]$$

where:
- $\alpha = [0.1, 0.2, 0.3, 0.4]$ progressively emphasizes full-resolution fine structural boundaries.
- $P_k$ denotes the predicted probability map at stage $k$.
- $Y$ is the ground-truth binary caries segmentation mask downsampled to the corresponding resolution.

### Composite Soft Dice Loss Formulation
Dental caries occupy less than $1\%$ of the total panoramic pixel space. To eliminate background dominance, Soft Dice loss directly optimizes spatial contour overlap:

$$\mathcal{L}_{\text{SoftDice}}(P, Y) = 1 - \frac{2 \sum_{i} P_i Y_i + \epsilon}{\sum_{i} P_i^2 + \sum_{i} Y_i^2 + \epsilon}$$

where $\epsilon = 10^{-6}$ provides numerical stability.

---

## 3. Semi-Supervised Learning & Teacher EMA Synchronization

Unlabeled radiographs are integrated via uncertainty-guided consistency regularization between a Student network and a slowly evolving Teacher network:

$$\mathcal{L}_{\text{unsup}} = \frac{1}{|U|} \sum_{x \in U} \exp(-\sigma^2(x)) \cdot \| f(x; \theta) - f(\tilde{x}; \theta_{\text{ema}}) \|_2^2$$

### Teacher Synchronization Formulation
To guarantee numerical stability and eliminate activation explosions, both parameters and floating-point BatchNorm running statistics are updated synchronously every step:

$$\theta_{\text{ema}} \leftarrow \beta \theta_{\text{ema}} + (1 - \beta) \theta$$
$$\mu_{\text{running\_ema}} \leftarrow \beta \mu_{\text{running\_ema}} + (1 - \beta) \mu_{\text{running}}$$
$$\sigma^2_{\text{running\_ema}} \leftarrow \beta \sigma^2_{\text{running\_ema}} + (1 - \beta) \sigma^2_{\text{running}}$$

with exponential moving average decay $\beta = 0.99$.

---

## 4. Monte Carlo Epistemic Uncertainty Estimation

Epistemic uncertainty arises from ambiguous clinical boundaries, cervical burnout radiolucencies, and restorative resin margins. During inference and unsupervised training, the framework executes $T = 8$ stochastic forward passes with active dropout ($p = 0.20$):

$$\mu(x) = \frac{1}{T} \sum_{t=1}^{T} \hat{y}_t(x)$$

$$\sigma^2(x) = \frac{1}{T} \sum_{t=1}^{T} \left( \hat{y}_t(x) - \mu(x) \right)^2$$

Pixel-wise variance $\sigma^2(x)$ acts as an exponential gating factor $\exp(-\sigma^2(x))$ to suppress noisy pseudo-labels in low-confidence regions.

---

## 5. Sliding-Window Panoramic Inference & Blending

1. **Resolution Standardization:** Full panoramic images are standardized to $768 \times 1536$ resolution with Contrast Limited Adaptive Histogram Equalization (CLAHE).
2. **Patch Decomposition:** 21 overlapping sub-patches of size $384 \times 384$ are extracted with a horizontal and vertical stride of 192 pixels.
3. **2D Gaussian Blending:** Reconstructed full-image probabilities are fused using Gaussian weight kernels centered on each patch to eliminate border artifacts.
4. **Binarization:** The final binary mask is generated using the calibrated operating threshold $\tau = 0.50$.
