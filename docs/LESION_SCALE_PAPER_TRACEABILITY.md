# LESION-SCALE PERFORMANCE ANALYSIS PAPER TRACEABILITY
**Project**: Standalone Dental Panoramic Caries Segmentation (MLUA)  
**Document**: `docs/LESION_SCALE_PAPER_TRACEABILITY.md`  
**Classification**: Paper-Aligned Lesion-Scale Evaluation Methodology on Dental Panoramic Radiographs  
**Status**: VERIFIED & DOCUMENTED

---

## 1. Overview & Theoretical Concept

In medical image segmentation, small and subtle lesions (e.g. incipient enamel caries) present significantly higher segmentation challenge compared to extensive cavitation due to low contrast, faint boundaries, and extreme class imbalance. The MLUA paper evaluates model robustness by partitioning targets across three canonical lesion-scale tiers.

> *Statement of Alignment*: **This is a paper-aligned lesion-scale evaluation methodology adapted for 2D panoramic dental radiographs.**

---

## 2. Lesion Size Tier Definitions

| Scale Tier | Pixel Area Criterion | Clinical / Topological Description |
| :--- | :--- | :--- |
| **`SMALL`** | $\text{Area} < 300\text{ pixels}$ | Incipient or shallow caries lesions, early enamel demineralization |
| **`MEDIUM`** | $300 \le \text{Area} \le 1000\text{ pixels}$ | Moderate dentinal caries, localized interproximal or occlusal lesions |
| **`LARGE`** | $\text{Area} > 1000\text{ pixels}$ | Extensive cavitation, deep dentinal or pulpal involvement lesions |

---

## 3. Methodological Comparison: Paper vs. Our Implementation

| Evaluation Dimension | Paper-Reported Methodology | Our Evaluation Framework | Classification & Notes |
| :--- | :--- | :--- | :--- |
| **Target Extraction** | 3D connected-component labeling on volumetric CT/MRI | 2D 8-connectivity labeling on panoramic radiograph masks | **Modality Adaptation**: Adapted for $768 \times 1536$ 2D panoramic caries annotations |
| **Primary Metric** | Stratified Dice Score per scale group | Stratified Dice, Sensitivity/Recall, Precision | **Paper-Aligned Primary Metrics** |
| **Lesion-Level Matching** | Slice / volumetric component metrics | Greedy 1-to-1 bipartite matching ($\text{IoU} \ge 0.10$) | **Additional Analysis Protocol** (Explicitly distinguished) |
| **Diagnostic Metrics** | Dice per volume tier | IoU, F1, Specificity, TP/FP/FN/TN confusion | **Extended Clinical Diagnostics** |
| **Case Aggregation** | Global average per scale group | Both Micro-Pixel Aggregation and Macro-Case Aggregation | **Dual-Level Statistical Reporting** |

---

## 4. Methodological Distinctions & Topological Limitations

1. **Connected Components vs. Clinical Lesions**:
   - In 2D panoramic radiograph ground truth, contiguous foreground pixel clusters extracted via connected component analysis represent annotated regions.
   - When adjacent proximal teeth have touching or overlapping caries annotations, connected components may merge multiple physical lesions into a single component.
   - *Rigor standard*: The pipeline treats connected components strictly as **annotated foreground components**, avoiding claims of absolute topological lesion individuation.
2. **Binary vs. Severity Annotations**:
   - The primary evaluation operates on **binary caries masks** ($0 = \text{background}, 1 = \text{caries}$).
3. **Class Imbalance & Specificity**:
   - In panoramic dental imaging ($768 \times 1536 = 1,179,648\text{ pixels}$ per image), background pixels constitute $\approx 99.8\%$ of total image area.
   - Consequently, True Negatives (TN) heavily dominate, causing raw Specificity and Accuracy to approach $\approx 0.999+$.
   - **Primary performance must strictly rely on Dice, Sensitivity/Recall, and Precision**.
