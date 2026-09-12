# EXP-MLUA-001 TRAINING HEALTH & TELEMETRY REPORT
**Generated**: 2026-09-09 14:52:23  
**Monitoring Target**: `EXP-MLUA-001_FULL_TRAINING_HISTORY.csv`  
**Overall Training Status**: **`INITIAL_FOREGROUND_EMERGENCE`**

---

## 1. Observed Facts (Empirical Metrics)

| Metric Property | Observed Value | Context / Benchmark |
| :--- | :--- | :--- |
| **Current Completed Epoch** | **Epoch 20 / 200** | Completed training iterations |
| **Best Validation Dice** | **0.1470** (14.70%) | Achieved at **Epoch 19** |
| **Best Validation IoU** | **0.0824** (8.24%) | Achieved at **Epoch 19** |
| **Best Validation Recall** | **0.1658** (16.58%) | Achieved at **Epoch 19** |
| **Current Validation Dice** | **0.0780** | Step delta: -0.0690 |
| **Current Validation Recall** | **0.0512** (5.12%) | Lesion detection sensitivity |
| **Current Validation Precision** | **0.3626** (36.26%) | Positive predictive value |
| **Current Specificity** | **0.9989** | True negative rate (background) |
| **Current Validation Loss** | **1.0127** | Step delta: -0.0062 |
| **Current Training Loss** | **0.6429** | Step delta: -0.0014 |
| **Current Learning Rate** | **0.0009095** | Polynomial decay active |
| **Peak Foreground Probability**| **0.8008** (if available) | Sigmoidal output peak |
| **Predicted Prevalence** | **0.001545** | Caries pixel area fraction |
| **Average Epoch Duration** | **16.62 minutes** | Across observed epochs |
| **Cumulative Runtime** | **`5:15:45`** | Wall-clock execution time |
| **Estimated Remaining Time** | **`2 days, 1:51:25`** (180 epochs) | Based on observed pace |

---

## 2. Heuristic Warnings & Signal Flags

| Heuristic Detector | Status / Flag | Criteria & Technical Evaluation |
| :--- | :---: | :--- |
| **Foreground Emergence** | **`DETECTED` (Epoch 15)** | Network transitioned from all-background to positive caries predictions ($P_{max} > 0.50$). |
| **All-Background Collapse** | **`NO (CLEARED)`** | Current model is actively detecting foreground lesions ($	ext{Recall} = 5.12\% > 0$). |
| **Extreme Imbalance Notice** | **`ACTIVE NOTICE`** | High specificity ($pprox 0.999$) is expected due to $>99.5\%$ background ratio and must not be used alone to judge quality. |
| **Overfitting Risk Signal** | **`NOMINAL`** | Epochs since best Dice: 1. Training loss is decreasing while validation loss is decreasing. |
| **Metric Instability Signal** | **`DETECTED`** | Single-epoch Dice drop exceeded threshold ($\Delta	ext{Dice} = -0.0690$). |

---

## 3. Scientific Interpretation & Trajectory Analysis

1. **Learning Regime Progression**:
   - *Epochs 2–11*: Early low-confidence warmup regime where logits remained sub-threshold ($	au=0.50$), producing nominal zero Dice with $pprox 1.0$ specificity.
   - *Epochs 12–15*: Foreground boundary emergence ($P_{max}$ scaled $0.41 	o 0.62$), initiating initial true positive detections.
   - *Epochs 16–20*: Active semi-supervised learning phase, reaching a peak validation Dice of **0.1470 (14.70%)** at Epoch 19 with peak foreground probability reaching **0.8978**.
2. **Current Trajectory**:
   - Model demonstrates healthy supervised and consistency loss convergence without NaN/Inf anomalies.
   - Training continues safely on schedule.
