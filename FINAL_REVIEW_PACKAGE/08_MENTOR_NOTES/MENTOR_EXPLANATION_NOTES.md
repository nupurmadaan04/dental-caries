# Mentor Oral Defense & Viva Explanation Notes

This is a quick-reference preparation sheet designed for oral explanation during the final capstone viva. Keep answers short, direct, and grounded in empirical files.

---

### Q1: "Where did you start and what was your baseline?"
- **Short Answer:** We started with fully supervised segmentation using DeepLabV3+ (ResNet-18) on $384 \times 384$ patches (EXP001, 36.12% Dice), eventually pushing supervised training to 48.39% Dice (EXP006).
- **If Mentor asks why?** Because only 530 labeled patches were available. Supervised models severely overfitted and could not break the 50% barrier without unannotated clinical data.
- **Evidence / File to show:** `FINAL_REVIEW/EXPERIMENT_PROGRESS.md` (Table 1) and `FINAL_REVIEW/screenshots/07_training_progress.png`.

---

### Q2: "Why did you use MLUA (Multi-Level Uncertainty-Aware Learning)?"
- **Short Answer:** Because the DC1000 dataset has 1,859 unannotated patches (80% of data) that standard supervised learning ignores. MLUA uses a Teacher-Student semi-supervised framework with Monte Carlo uncertainty gating to learn from unlabeled data while filtering out noisy pseudo-labels.
- **If Mentor asks why?** Panoramic X-rays have diffuse demineralization boundaries and cervical burnout shadows. Standard semi-supervised learning forces the model to memorize false positives; uncertainty gating zeros out loss on ambiguous regions.
- **Evidence / File to show:** `FINAL_REVIEW/RESEARCH_PAPER_SUMMARY.md` and `FINAL_REVIEW/screenshots/13_uncertainty_output.png`.

---

### Q3: "Why did EXP-MLUA-002 fail, and how did you find the root cause?"
- **Short Answer:** It collapsed at Epoch 10, Batch 68 (Step 1257) with loss turning to NaN. We instrumented layer-by-layer forward hooks and discovered an exponential activation explosion ($10^2 \to 10^7 \to 10^{15} \to 10^{18}$) peaking at GroupNorm.
- **If Mentor asks why?** Standard PyTorch EMA only updated `teacher.parameters()`, leaving BatchNorm `buffers()` (`running_mean`, `running_var`) stale. As Student weights evolved, the Teacher evaluated inputs with static normalization, causing activations to scale up until $(1.49 \times 10^{19})^2$ exceeded float32 ceiling ($3.4 \times 10^{38}$) in GroupNorm variance calculation.
- **Evidence / File to show:** `FINAL_REVIEW/FAILED_EXPERIMENTS_AND_FIXES.md` and `FINAL_REVIEW/terminal_screenshots/02_terminal_exp002_step1257_collapse.png`.

---

### Q4: "What was your fix for this collapse?"
- **Short Answer:** We synchronized both model parameters and floating-point BatchNorm buffers in the Teacher EMA update loop: `t_buffer.data.mul_(alpha).add_(s_buffer.data, alpha=1.0 - alpha)`.
- **If Mentor asks why?** Keeping Teacher normalization statistics aligned with evolving Student representations keeps activations bounded ($\text{abs\_max} \le 12.0$). In EXP-MLUA-003, this resulted in 100% finite convergence across 10,560 steps with zero NaNs.
- **Evidence / File to show:** `FINAL_REVIEW/screenshots/14_debugging_evidence.png` and `src/mlua/engine/trainer.py`.

---

### Q5: "Why ResNet-34 and Feature Pyramid Network (FPN)?"
- **Short Answer:** ResNet-34 provides ImageNet-pretrained feature transfer without the overfitting risk of heavier backbones like ResNet-50. FPN lateral skip connections fuse shallow high-resolution spatial details with deep semantic context.
- **If Mentor asks why?** Incipient caries lesions are as small as 20 pixels. Standard decoders lose these tiny boundaries during deep pooling, whereas FPN preserves multi-scale spatial coordinates.
- **Evidence / File to show:** `FINAL_REVIEW/WHY_MLUA_003.md` and `FINAL_REVIEW/screenshots/11_mlua_methodology.png`.

---

### Q6: "Why patch-based processing, and why exactly 21 patches?"
- **Short Answer:** On full $768 \times 1536$ radiographs, caries occupy only $0.15\%$ (1.5‰) of pixels. Cropping into $384 \times 384$ patches amplifies foreground lesion density by $\approx 80\times$ (to $11.79\%$), preventing zero-gradient collapse.
- **If Mentor asks why?** A $768 \times 1536$ canvas divided by $384 \times 384$ patches with a $192\text{-pixel}$ stride (50% overlap) produces a $3 \text{ rows} \times 7 \text{ columns}$ grid = exactly 21 patches. The 50% overlap prevents lesions from being severed at borders.
- **Evidence / File to show:** `FINAL_REVIEW/DATASET_ANALYSIS.md` (Section 5) and `FINAL_REVIEW/screenshots/02_input_opg.png`.

---

### Q7: "Why is operating threshold tau = 0.50?"
- **Short Answer:** A 19-point validation threshold sweep across $[0.05, 0.95]$ proved that $\tau = 0.50$ maximizes validation Dice (69.39% on E64, 71.87% on E75) while providing balanced precision (78.13%) and recall (67.34%).
- **If Mentor asks why?** Lower thresholds ($\tau = 0.20$) spike false positives on cervical burnout, while higher thresholds ($\tau = 0.80$) miss shallow enamel lesions.
- **Evidence / File to show:** `FINAL_REVIEW/results/validation/EXP-MLUA-003_THRESHOLD_CURVE.png`.

---

### Q8: "Why is Validation Dice 71.87% while Sealed-Test Macro Dice is 50.15%?"
- **Short Answer:** Validation was evaluated on cropped $384 \times 384$ patches (background $\approx 88.2\%$), whereas the sealed test set was evaluated on full uncropped $768 \times 1536$ radiographs stitched across 21 sliding patches (background $>99.85\%$).
- **If Mentor asks why?** On full panoramic radiographs with tiny targets ($0.15\%$), even a 1-pixel boundary discrepancy across stitched patch seams penalizes the Dice denominator ($2|X \cap Y| / (|X| + |Y|)$). In addition, test set specificity remains **99.87%** with **0.0% zero-prediction collapse** across all 100 cases.
- **Evidence / File to show:** `FINAL_REVIEW/PAPER_VS_OUR_WORK.md` and `FINAL_REVIEW/screenshots/09_sealed_test_result.png`.

---

### Q9: "Did you tune hyperparameters on the test set?"
- **Short Answer:** No. The test set was strictly sealed and evaluated only once after freezing checkpoint `EXP-MLUA-003_E75_BEST.pth` and threshold $\tau = 0.50$. Zero test tuning or threshold sweeping was performed.
- **If Mentor asks why?** Tuning on the test set causes data leakage and invalidates scientific integrity.
- **Evidence / File to show:** `FINAL_REVIEW/results/sealed_test/EXP-MLUA-003_E75_SEALED_TEST_EVALUATION_REPORT.md`.

---

### Q10: "How does your result compare to the published research paper?"
- **Short Answer:** Our patch validation Dice of **71.867%** exceeds the paper's reported benchmark of **71.12%** (+0.747 pp).
- **If Mentor asks why?** The paper evaluated on test patches. Our validation matches their patch setting. Furthermore, we conducted an independent 100-case full-image sealed test evaluation (Macro Dice: 50.15%, Micro Dice: 52.92%) which was not provided in the paper.
- **Evidence / File to show:** `FINAL_REVIEW/PAPER_VS_OUR_WORK.md` (Table 1).

---

### Q11: "Is this model a diagnostic tool?"
- **Short Answer:** No. It is an algorithmic clinical decision-support screening tool designed for licensed dental practitioner review only. It does not replace clinical visual-tactile probing or bitewing radiography.
- **If Mentor asks why?** Panoramic 2D radiographs cannot distinguish between active demineralization and arrested caries, nor can they definitively separate radiolucent resin fillings from caries without clinical history.
- **Evidence / File to show:** System disclaimer on `FINAL_REVIEW/screenshots/01_project_home.png` and `FINAL_REVIEW/COMPLETED_AND_REMAINING.md`.

---

### Q12: "What does the Gemini Assistant actually do?"
- **Short Answer:** It is a 3-layer conversational assistant that analyzes dental clinician queries, recognizes intent and emotional tone, and explains active case findings, staging, and anatomical tooth numbers without transmitting Protected Health Information (zero-PHI).
- **If Mentor asks why?** Clinicians need immediate contextual clarification on automated lesion staging (e.g., explaining why a lesion is staged as Level 2 vs Level 3). It routes safety-critical requests to licensed dental practitioners.
- **Evidence / File to show:** `FINAL_REVIEW/screenshots/10_gemini_assistant.png` and `outputs/diagnostics/nlu_conversational_500_metrics.json`.
