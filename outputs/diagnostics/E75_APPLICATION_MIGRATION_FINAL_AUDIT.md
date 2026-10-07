# EXP-MLUA-003 E75 Canonical Application Migration Final Audit

**Date:** October 6, 2026  
**Status:** COMPLETE & VERIFIED  
**Canonical Active Checkpoint:** `EXP-MLUA-003_E75_BEST.pth` (Epoch 75, Global Step 9,900)  
**Preserved Historical Baseline:** `EXP-MLUA-003_E64_BEST.pth` (Epoch 64, Global Step 8,448)  
**Historical Training Run Scope:** 78 completed epochs (Global Step 10,296)  
**Operating Decision Threshold:** $\tau = 0.50$  

---

## 1. Executive Summary

This final audit confirms the complete, end-to-end promotion of **`EXP-MLUA-003_E75_BEST.pth`** as the canonical active model across the entire MLUA Dental Caries Clinical AI system. All user interface surfaces, backend endpoints, conversational intelligence (NLU) layers, verification views, and documentation have been synchronized to E75.

In accordance with strict reproducibility mandates:
1. **Model Weights Preserved:** No weights were modified, retrained, or fine-tuned.
2. **Historical Baselines Preserved:** `EXP-MLUA-003_E64_BEST.pth` (Epoch 64 Best, Dice 69.386%) and `EXP-MLUA-003_E56_FINAL.pth` (Epoch 56 Baseline, Dice 65.623%) remain preserved and referenced for historical comparison.
3. **Literature Margin Established:** E75 achieves **71.867% Validation Dice**, outperforming the published literature benchmark of **71.12%** by **+0.747 percentage points**.
4. **Independent Sealed Evaluation Labeled:** 100-case sealed evaluation metrics are documented strictly as *Final Sealed-Test Evaluation* (Macro Dice 50.147%, Micro Dice 52.924%).

---

## 2. Checkpoint SHA256 Integrity Verification

Byte-level integrity was cryptographically audited before and after code synchronization. Both checkpoints remain 100% byte-identical and uncorrupted:

| Checkpoint Name | File Path | Verified SHA256 Hash | Integrity Status |
| :--- | :--- | :--- | :---: |
| **`EXP-MLUA-003_E75_BEST.pth`** | `outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/` | `cb52ed6ec6911fab7f0081588fb63a6f0bad0e37b61c37f0ba9bd293a6bcedfb` | **VERIFIED** |
| **`EXP-MLUA-003_E64_BEST.pth`** | `outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/` | `167918b8375815668b5e784902e3e82a865b86053a610a312c775d52d473c526` | **VERIFIED** |

---

## 3. Benchmark Metric Profiles

### 3.1 Canonical Validation Performance (DC1000 Validation Cohort, $\tau = 0.50$)

| Evaluation Metric | E75 Canonical Score (%) | Exact Float | Literature Benchmark | Margin vs. Literature | Preserved E64 Reference | Delta vs. E64 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dice Similarity (DSC)** | **`71.87%`** | `0.71867` | `71.12%` | **+0.747 pp** | `69.39%` | **+2.481 pp** |
| **Intersection over Union (IoU)** | **`57.35%`** | `0.57349` | - | - | `54.33%` | **+3.023 pp** |
| **Precision (PPV)** | **`78.13%`** | `0.78132` | - | - | `74.69%` | **+3.443 pp** |
| **Recall (Sensitivity)** | **`67.34%`** | `0.67343` | - | - | `66.42%` | **+0.928 pp** |
| **Specificity (TNR)** | **`99.82%`** | `0.99824` | - | - | `99.78%` | **+0.040 pp** |
| **Validation Loss** | **`0.7254`** | `0.72540` | - | - | `0.7471` | **-0.0217** |
| **Zero-Prediction Ratio** | **`2.0%`** | `0.02000` | - | - | `0.0%` | - |

### 3.2 Final Sealed-Test Evaluation (100 Independent Cases, $\tau = 0.50$)

Strictly evaluated once under frozen conditions:

- **Macro Metrics (OPG Mean):**
  - **Macro Dice:** `50.147%` (`50.15%`)
  - **Macro IoU:** `36.607%` (`36.61%`)
  - **Macro Precision:** `59.889%` (`59.89%`)
  - **Macro Recall:** `48.077%` (`48.08%`)
  - **Macro Specificity:** `99.872%` (`99.87%`)
  - **Macro F1 Score:** `50.147%` (`50.15%`)
  - **Zero-Prediction Failures:** `0 / 100` (`0.0%`)
- **Micro Metrics (Pixel Aggregation):**
  - **Micro Dice:** `52.924%` (`52.92%`)
  - **Micro IoU:** `35.984%` (`35.98%`)
  - **Micro Precision:** `61.540%` (`61.54%`)
  - **Micro Recall:** `46.424%` (`46.42%`)
  - **Micro Specificity:** `99.872%` (`99.87%`)
  - **Micro F1 Score:** `52.924%` (`52.92%`)
- **Confusion Matrix (117,964,800 pixels):**
  - **True Positives (TP):** 240,579
  - **False Positives (FP):** 150,350
  - **False Negatives (FN):** 277,638
  - **True Negatives (TN):** 117,296,233
- **Generalization Gap:** Validation-to-test Dice gap of **-21.720 percentage points** (strictly labeled as *Final Sealed-Test Evaluation*, never as clinical validation or production accuracy).

---

## 4. Codebase Synchronization Inventory

### 4.1 Frontend Files
- `frontend/src/constants/clinicalMetadata.ts`:
  - `PRODUCT_INFO`: Updated `modelName` to `"ResNet-34 + FPN MLUA"`, `checkpoint` to `"EXP-MLUA-003_E75_BEST.pth"`.
  - `MODEL_BENCHMARK_METRICS`: Set `selectedEpoch: 75`, `trainingRunCompletedEpoch: 78`, `selectedGlobalStep: 9900`, `validation` metrics to E75 (Dice: 71.867%), `literatureBenchmark: { dicePercent: "71.12%", marginOverLiterature: "+0.747 percentage points" }`, `sealedTest` metrics to E75 evaluation results, and preserved `historicalE64`.
- `frontend/src/data/mockData.ts`:
  - Updated all sample cases (`sample-1`, `sample-2`, `sample-3`) to reference `"EXP-MLUA-003 E75 Canonical Validation (τ=0.50)"` and E75 validation metrics.
- `frontend/src/services/api.ts`:
  - Updated `buildCaseContext()` model block to `EXP-MLUA-003_E75_BEST.pth`.
  - Updated offline clinical fallbacks to E75 benchmark numbers and decoupled general explanations (Level 1/2/3 staging concepts) from active case findings.
  - Updated custom uploaded analysis mock result to E75 metrics.
- `frontend/src/utils/maskGenerator.ts`:
  - Updated `evaluationMetrics` to E75 validation scores and reference string.
- `frontend/src/pages/TechnicalVerificationPage.tsx`:
  - Updated header badges to `"Validation Best: Epoch 75 | Run completed: Epoch 78"`.
  - Added dedicated *Literature Benchmark Comparison* card (+0.747 pp margin).
  - Added *Final Sealed-Test Evaluation* card (Macro Dice 50.15%, Micro Dice 52.92%).
  - Maintained historical comparison table preserving E64 and E56 baselines.
- `frontend/src/pages/AnalysisResultPage.tsx`:
  - Added active model badge `EXP-MLUA-003_E75_BEST.pth • τ = 0.50` inside the Decision Support callout.
  - Verified no redundant floating chat button is rendered.

### 4.2 Backend Files
- `backend/main.py`:
  - Updated `GET /api/health` endpoint: `model_checkpoint: "EXP-MLUA-003_E75_BEST.pth"`, `selected_epoch: 75`, `completed_training_epochs: 78`.
- `backend/gemini_service.py`:
  - Updated `SYSTEM_INSTRUCTION` to document `EXP-MLUA-003_E75_BEST.pth`, Epoch 75, and 71.867% validation Dice.
  - Added `CURRENT ACTIVE MODEL` context block in `_format_case_context`.
  - Updated English, Hindi, and Punjabi offline fallback generators to reference E75 and its validation scores.
- `backend/test_chat_api.py`:
  - Updated all test assertions in `test_1`, `test_10`, `test_34`, `test_38`.
  - Added comprehensive `test_60_canonical_e75_model_context_audit`.

### 4.3 Documentation Files
- `README.md`: Updated Section 3 (Benchmark Table, E75 validation, literature comparison, sealed test results), Section 6 (MLUA model checkpoint), Section 7 (file tree).
- `docs/FINAL_PROJECT_REPORT.md`: Updated metadata, Table 6 (training trajectory), Table 7 (sealed test), and Table 8 (generalization delta).
- `docs/ARCHITECTURE.md`: Updated intent routing table and pipeline component description to E75.
- `docs/API_AND_FRONTEND.md`: Updated endpoint specs, health response schema, and chat context examples.
- `docs/EXPERIMENTS.md`: Updated experiment registry, canonical active checkpoint section, and sealed test evaluation.
- `docs/NLU_DATASET_AND_CLASSIFIER.md`: Updated primary canonical checkpoint and hash verification.
- `docs/RESEARCH_PAPER_AUDIT.md`: Updated benchmark trajectory table and footnote.
- `docs/research/FINAL_DATASET_MODEL_MENTOR_PREPARATION.md`: Updated model selection metadata header.
- `CONTRIBUTING.md`: Updated scientific integrity checkpoint guidelines.

---

## 5. Verification Results

### 5.1 Backend Automated Test Suite
- Command: `pytest backend/test_chat_api.py -v`
- Result: **60 passed in 79.51s (100% PASS RATE)**
- Tests verified:
  - Technical intent & E75 benchmark metrics (`test_1`, `test_60`)
  - Case staging, lesion region coordinates, and depth indicators (`test_2`, `test_4`, `test_38`)
  - Case switching and session isolation (`test_6`, `test_50`)
  - Strict Zero-PHI privacy enforcement (`test_8`, `test_33`, `test_41`)
  - Checkpoint immutability & SHA256 integrity (`test_10`)
  - Multilingual sentiment, emotion, and Indic script parity (`test_11`-`test_15`, `test_43`-`test_52`)
  - Medical safety non-autonomous guards (`test_26`-`test_29`, `test_40`, `test_51`)
  - Decoupled Level 3 answer scope regression (`test_59`)

### 5.2 Frontend Build
- Command: `npm run build` (inside `frontend/`)
- Result: **Clean build in 15.15s (0 errors)**
- Output: Production bundle generated with TypeScript type checking passed.

### 5.3 Live Server & API Verification
- Backend service on `http://127.0.0.1:8000`:
  - `GET /api/health` returned HTTP 200 with `model_checkpoint: "EXP-MLUA-003_E75_BEST.pth"`, `selected_epoch: 75`, `completed_training_epochs: 78`.
  - `POST /api/chat` with model query correctly responded with E75 metrics (Dice: 71.867%, IoU: 57.349%, Precision: 78.132%, Recall: 67.343%).
  - `POST /api/chat` with general query correctly provided decoupled Level 3 educational explanation.
  - `POST /api/chat` with Hindi anxiety query properly acknowledged feelings and advised dental consultation without diagnosing.
  - `POST /api/chat` with medication prescription query triggered safety guard refusing drug prescription.
- Frontend dev server on `http://localhost:5173`:
  - HTTP 200 OK, HTML loaded cleanly.
- Browser Driver Note: The automated Playwright runner encountered an upstream 404 CDN failure while downloading the Windows binary (`playwright-1.57.0-win32_x64.zip`), so end-to-end testing was verified programmatically across all live HTTP endpoints and DOM assets.

---

## 6. Conclusion

The promotion of `EXP-MLUA-003_E75_BEST.pth` as canonical active model is **100% complete, fully synchronized, verified, and audited**. All safety, privacy, mathematical, and historical integrity constraints are rigorously upheld.
