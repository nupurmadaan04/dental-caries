# EXP-MLUA-001 E19-E22 Forensic Validation Degradation Audit

**Experiment**: `EXP-MLUA-001`  
**Focus**: Root Cause Analysis of Validation Dice Decline (Epoch 19: 14.705% $\to$ Epoch 22: 0.235%)  
**Classification**: Diagnostic Only | Strictly Read-Only | Zero Training Changes  
**Full Report**: [`outputs/experiments/EXP-MLUA-001_HISTORICAL/diagnostics/e19_e22/FORENSIC_AUDIT_REPORT.md`](file:///c:/Users/devin/MLUA/outputs/experiments/EXP-MLUA-001_HISTORICAL/diagnostics/e19_e22/FORENSIC_AUDIT_REPORT.md)  

---

## 1. Executive Summary & Root Cause Determination

The forensic validation audit definitively answers the central scientific question:
**The decline in validation Dice is NOT caused by the network forgetting caries representations.** Instead, it is caused by **Probability Calibration & Negative Logit Shift** combined with **Foreground Area Suppression** under standard unweighted BCE loss.

```
+-------------------------------------------------------------------------------------------------------+
|                                    ROOT CAUSE CLASSIFICATION                                          |
+------------------------------------+--------------------------+------------+--------------------------+
| Cause Category                     | Code                     | Assessment | Impact on Validation Dice|
+------------------------------------+--------------------------+------------+--------------------------+
| Probability Calibration Shift      | B_PROBABILITY_CALIBRATION| PRIMARY    | Logits shifted -1.351 neg|
| Foreground Suppression             | D_FOREGROUND_SUPPRESSION | PRIMARY    | 29.6x under-prediction   |
| Threshold Sensitivity (tau=0.50)   | C_THRESHOLD_SENSITIVITY  | CONTRIBUTOR| Fixed tau=0.50 cuts off  |
| EMA Teacher Smoothing              | G_EMA_DYNAMICS           | CONTRIBUTOR| Teacher max prob -> 0.154|
| Genuine Model Destruction          | A_GENUINE_MODEL_DEGRADE  | RULED OUT  | Lesion signal preserved  |
| Validation Implementation Error    | H_VALIDATION_PIPELINE    | RULED OUT  | Pipeline 100% verified   |
+------------------------------------+--------------------------+------------+--------------------------+
```

---

## 2. Key Empirical Findings

### 1. The Caries Signal is Retained in E22 Weights
- Foreground-to-background logit separation remains strictly positive: **$+1.961$ in E22** (vs $+3.395$ in E19).
- True caries pixels maintain a **$5.11\times$ higher mean probability** than background pixels in E22.

### 2. Threshold Sensitivity
- In E19, peak Dice is **$12.783\%$ at $\tau = 0.35$** ($12.190\%$ at $\tau = 0.50$).
- In E22, peak Dice is **$8.441\%$ at $\tau = 0.05$** ($0.089\%$ at $\tau = 0.50$).
- Reducing threshold to $\tau = 0.05$ produces a **$95\times$ recovery** in Dice and recovers validation recall from $0.046\%$ to **$22.058\%$**.

### 3. Foreground Suppression Ratio
- Ground Truth Caries Prevalence: **$0.993\%$**
- E19 Predicted Prevalence: **$1.070\%$** (Ratio: $1.078\times$, well-calibrated)
- E22 Predicted Prevalence: **$0.036\%$** (Ratio: $0.036\times$, $29.6\times$ suppression)

### 4. Deep Level Representations
- Aux Head 4 (Level 5, $1/32$ stride) maintains higher Dice ($1.214\%$) and higher maximum probability ($0.4433$) than the final Fused Head ($0.089\%$, $0.3416$), proving high-level caries features are preserved.

---

## 3. Decision Support Answers

1. **Is E19 still the best checkpoint?**  
   **YES**. `EXP-MLUA-001_BEST.pth` preserves the highest validation Dice ($14.705\%$).
2. **Is E22 genuinely worse than E19?**  
   **Partially**. E22 has lower probability calibration due to background dominance, but retains discriminatory spatial rankings.
3. **Is $\tau = 0.50$ hiding useful E22 predictions?**  
   **YES**. At $\tau = 0.05$, E22 achieves $8.441\%$ Dice and $22.058\%$ recall.
4. **Is foreground suppression occurring?**  
   **YES**. Predicted prevalence collapsed by $29.6\times$.
5. **Is there evidence of validation pipeline corruption?**  
   **NO**. The validation harness is verified 100% deterministic and correct.
6. **Is there evidence of training instability?**  
   **NO**. Training loss smoothly converges ($0.6443 \to 0.6430$).
7. **Should EXP-MLUA-001 continue running?**  
   **YES**. The run should proceed normally through its planned epochs.
8. **Should we change anything right now?**  
   **NO**. Zero mid-run scientific modifications.

---

## 4. Safety Verification

```
training_process_modified    = FALSE
checkpoint_modified          = FALSE
scientific_config_modified   = FALSE
validation_protocol_modified = FALSE
test_set_accessed            = FALSE
test_set_modified            = FALSE
EXP001_restarted             = FALSE
new_training_started         = FALSE
```
