# Research Paper Summary for Mentor Review

## 1. Paper Title
**"Multi-level uncertainty aware learning for semi-supervised dental panoramic caries segmentation"**

---

## 2. Year & Publication
- **Year:** 2023
- **Journal:** *Neurocomputing*, Volume 540, Article 126208, Elsevier.
- **Authors:** Xianyun Wang, Sizhe Gao, Kaisheng Jiang, Huicong Zhang, Linhong Wang, Feng Chen, Jun Yu, Fan Yang.
- **DOI:** [10.1016/j.neucom.2023.03.069](https://doi.org/10.1016/j.neucom.2023.03.069).

---

## 3. Problem Addressed
Accurate, automated pixel-level dental caries segmentation on panoramic radiographs (OPGs) under severe clinical annotation scarcity, high background imbalance ($>99.8\%$), and ambiguous demineralization boundaries.

---

## 4. What MLUA Means
**MLUA** stands for **Multi-Level Uncertainty-Aware** learning:
- **Multi-Level:** Predictions are extracted from 4 multi-resolution auxiliary decoder stages ($1/8, 1/4, 1/2, 1/1$) in addition to the main fused head.
- **Uncertainty-Aware:** Epistemic uncertainty is estimated across multiple stochastic Monte Carlo passes to filter out ambiguous pseudo-labels before computing consistency loss.

---

## 5. Basic Architecture
- **Backbone Encoder:** ResNet-34 (pretrained on ImageNet), extracting hierarchical features at stages $C_1, C_2, C_3, C_4, C_5$.
- **Decoder:** Feature Pyramid Network (FPN) with top-down lateral connections fusing coarse semantic representations with fine spatial details across 4 scales ($P_2, P_3, P_4, P_5$, 256 channels each).
- **Deep Supervision Heads:** 4 auxiliary $1 \times 1$ conv heads with weights $\alpha = [0.1, 0.2, 0.3, 0.4]$, plus a final fused prediction head.

---

## 6. Teacher / Student Framework Idea
- **Student Network:** Trained with standard gradient backpropagation on labeled data (supervised loss: BCE + Soft Dice) and on unlabeled data (consistency loss).
- **Teacher Network:** Evaluates perturbed unlabeled patches. Its weights are NOT updated by gradient descent; instead, they are updated smoothly as an Exponential Moving Average (EMA, decay $\theta = 0.99$) of the Student weights:
  $$\theta_{\text{Teacher}} \leftarrow \beta \theta_{\text{Teacher}} + (1 - \beta) \theta_{\text{Student}}$$
- The Teacher provides stable target predictions for the Student to match.

---

## 7. Uncertainty Estimation Idea
- The model performs $T=8$ stochastic forward passes on perturbed unlabeled patches using active dropout ($p=0.20$) and subtle Gaussian input noise.
- Across 8 passes and 5 prediction heads ($40$ forward outputs), the pixel-wise prediction variance / entropy is calculated.
- Pixels with high variance represent high epistemic uncertainty.
- A dynamic threshold creates a binary certainty mask ($m_{\text{certain}}$) that zeros out consistency loss on uncertain pixels.

---

## 8. Why Uncertainty Is Useful for This Dataset
Panoramic radiographs contain natural anatomical shadows such as **cervical burnout**, overlap at interproximal spaces, and radiolucent fillings. Without uncertainty gating, the Teacher generates noisy predictions on these shadows, and consistency loss forces the Student to reinforce false positives. Uncertainty gating suppresses loss on these ambiguous pixels so the model only learns from confident dental tissue.

---

## 9. Paper's Evaluation Setup
- Evaluated on test patches from the DC1000 dataset using a 20% labeled / 80% unlabeled partitioning protocol.
- Metric reported: Mean Dice Similarity Coefficient (DSC) across test patches, stratified by lesion size ($DICE_s < 300\text{ px}$, $DICE_m 300-1000\text{ px}$, $DICE_l > 1000\text{ px}$).

---

## 10. Paper's Reported Result
- **Overall Reported Mean Dice:** **71.12%** (on test patches under the 20% labeled setting).
- **Breakdown by Lesion Size:**
  - Shallow ($DICE_s$): $57.06\%$
  - Middle ($DICE_m$): $75.82\%$
  - Deep ($DICE_l$): $84.28\%$

---

## 11. What We Followed
- ResNet-34 encoder + FPN decoder architecture.
- 4 multi-scale auxiliary heads with deep supervision loss weights $\alpha = [0.1, 0.2, 0.3, 0.4]$.
- Teacher-Student Mean Teacher framework with EMA updates ($\theta = 0.99$).
- Monte Carlo uncertainty gating ($T=8$ stochastic passes) and dynamic certainty masking.
- DC1000 dataset partitioning (20% labeled = 530 patches; 80% unlabeled = 1,859 patches).
- Patch size $384 \times 384$ pixels with $192\text{ px}$ stride.

---

## 12. What We Changed / Added in Our Implementation

| Aspect | Research Paper | Our Implementation | Reason |
|---|---|---|---|
| **Teacher EMA Synchronization** | Parameter weights only | **Parameters AND BatchNorm running buffers** | Stale buffers caused $10^{18}$ activation explosion and NaN collapse in EXP-002; buffer sync fixed this completely. |
| **Full OPG Reconstruction** | Simple patch averaging | **21-patch sliding window with 2D Gaussian spatial probability blending** | Eliminates seam edge artifacts along overlapping patch boundaries on full $768 \times 1536$ radiographs. |
| **Evaluation Scope** | Patch-level evaluation | **Patch validation (E75: 71.87%) AND full-image 100-case sealed test evaluation (Macro: 50.15%, Micro: 52.92%)** | Provides real-world full-canvas evaluation across entire dental arches ($117.96\text{M}$ pixels). |
| **System Integration** | Standalone research script | **Production FastAPI backend + React clinical UI + Case-aware AI Assistant** | Creates an interactive clinical decision-support tool for dental review. |

> **Fair Comparison Note for Mentor:**  
> Our validation score of **71.867%** was measured on $384 \times 384$ validation patches, which closely matches the paper's patch evaluation setup (71.12%). Our sealed-test result of **50.147% Macro Dice / 52.924% Micro Dice** was evaluated on full uncropped $768 \times 1536$ panoramic images where background is $>99.85\%$. These two evaluation conditions are mathematically different and must not be conflated.
