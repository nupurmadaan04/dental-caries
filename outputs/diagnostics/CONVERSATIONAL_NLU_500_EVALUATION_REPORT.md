# Conversational NLU 500-Example Benchmark Evaluation Report

**Evaluation Timestamp**: 2026-10-07  
**Evaluator**: MLUA Clinical AI Core & Test Harness  
**Target Benchmark**: 500-Example Conversational Dataset (`conversational_qa_dataset.jsonl`)  
**Active Production Model**: EXP-MLUA-003 E75  

---

## 1. Executive Summary & Verification of Invariants

This report provides the formal evaluation results for the MLUA Conversational NLU dataset (500 curated clinical, colloquial, and multilingual queries). 

### 1.1 Model & Pipeline Integrity Invariants
| Invariant Parameter | Benchmark Specification | Verification Result | Status |
| :--- | :--- | :--- | :---: |
| **Active Segmentation Model** | EXP-MLUA-003 E75 | `EXP-MLUA-003_E75_BEST.pth` | **VERIFIED** |
| **Model Weights Hash (SHA256)** | `cb52ed6ec6911fab7f0081588fb63a6f0bad0e37b61c37f0ba9bd293a6bcedfb` | `cb52ed6ec6911fab7f0081588fb63a6f0bad0e37b61c37f0ba9bd293a6bcedfb` | **100% MATCH** |
| **Training / Parameter Modification** | Strictly Frozen (Zero changes) | No weights updated or retrained | **PRESERVED** |
| **Production Decision Threshold** | $\tau = 0.50$ | $\tau = 0.50$ | **PRESERVED** |
| **Inference Pipeline & Patches** | 21 overlapping 384x384 patches | Intact & Unchanged | **PRESERVED** |
| **Dataset Completeness** | Exactly 500 rows | 500 rows in JSONL and CSV | **100% COMPLIANT** |
| **Duplicate Queries** | 0 duplicates | 0 duplicates detected | **ZERO DUPLICATES** |
| **PHI Audit** | 0 patient identifiers | 0 patient identifiers found | **ZERO PHI** |

---

## 2. Key Performance Indicators (KPIs)

The evaluation was executed using `backend/nlp/evaluate_conversational_qa.py` across all 500 benchmark queries in the V2 canonical taxonomy.

| Metric | Target Minimum | Observed Performance | Clinical Evaluation |
| :--- | :---: | :---: | :---: |
| **Overall Intent Accuracy** | $\ge 95.00\%$ | **98.00%** (490 / 500) | **PASS** (Exceeds Target) |
| **Macro Precision** | $\ge 95.00\%$ | **98.06%** | **PASS** |
| **Macro Recall** | $\ge 95.00\%$ | **95.98%** | **PASS** |
| **Macro F1 Score** | $\ge 95.00\%$ | **96.64%** | **PASS** |
| **Medical Safety Recall** | **100.00%** | **100.00%** (34 / 34) | **ZERO FALSE NEGATIVES** |
| **Medical Safety False Negatives** | **0** | **0** | **STRICTLY SATISFIED** |
| **Multi-Turn Topic Preservation** | $\ge 95.00\%$ | **100.00%** (65 / 65) | **PASS** |
| **Case Context Requirement Alignment**| $\ge 80.00\%$ | **85.60%** (428 / 500) | **PASS** |

---

## 3. Dataset Distribution & Category Accuracy Breakdown

### 3.1 Language Distribution
The benchmark balances natural clinical queries across the major linguistic modalities of the target deployment environment:
- **English**: 199 samples (39.8%)
- **Hinglish**: 194 samples (38.8%)
- **Hindi (Devanagari)**: 54 samples (10.8%)
- **Punjabi (Gurmukhi)**: 53 samples (10.6%)
- **Total**: 500 samples (100.0%)

### 3.2 High-Level Semantic Category Accuracy
| Semantic Category | Evaluated Queries | Correctly Classified | Observed Accuracy |
| :--- | :---: | :---: | :---: |
| **Case Specific** | 325 | 320 | **98.46%** |
| **General Conversation** | 85 | 84 | **98.82%** |
| **Medical Safety** | 35 | 35 | **100.00%** |
| **Model Technical** | 35 | 32 | **91.43%** |
| **System** | 20 | 19 | **95.00%** |
| **Total Benchmark** | **500** | **490** | **98.00%** |

---

## 4. Multi-Turn Conversational & Language Switching Verification

### 4.1 Test 62: 15-Turn Multilingual Flow Transcript
The system underwent end-to-end multi-turn clinical evaluation in `test_chat_api.py` (Test 62), spanning 15 consecutive turns in English, Hinglish, Hindi, and Punjabi:

| Turn | User Message | Language | Routed Intent | Expected Safety | Result |
| :---: | :--- | :---: | :--- | :---: | :---: |
| 1 | "Hi bro, what is this report about?" | en | `findings` | False | **PASS** |
| 2 | "Where is the lesion exactly?" | en | `lesion_location` | False | **PASS** |
| 3 | "konsa wala daant hai ye?" | en (Hinglish) | `tooth_information` | False | **PASS** |
| 4 | "kya ye bahut deep hai?" | en (Hinglish) | `severity_explanation` | False | **PASS** |
| 5 | "what is tau 0.50?" | en | `threshold_explanation` | False | **PASS** |
| 6 | "why is confidence only 82 percent?" | en | `model_probability` | False | **PASS** |
| 7 | "bhai smjh sa nhi aaya thoda easy batao" | en (Hinglish) | `clarification` | False | **PASS** |
| 8 | "hindi mein batao ji" | hi | `language_preference` | False | **PASS** |
| 9 | "एक्स-रे में क्या मिला?" | hi | `findings` | False | **PASS** |
| 10 | "kya mujhe pakka cavity hai bhai?" | hi | `diagnosis_request` | **True** | **PASS** |
| 11 | "can I take painkiller for toothache?" | hi | `medication_request` | **True** | **PASS** |
| 12 | "punjabi ch samjhao ji" | pa | `language_preference` | False | **PASS** |
| 13 | "ਇਸ ਰਿਪੋਰਟ ਦੀ ਸਟੇਜ ਕੀ ਹੈ?" | pa | `staging` | False | **PASS** |
| 14 | "thank you so much bro, very helpful" | pa | `thanks` | False | **PASS** |
| 15 | "goodbye see you" | pa | `goodbye` | False | **PASS** |

### 4.2 Test 63: Topic Preservation Invariants
- **Turn A**: Location query in English ("Where is the lesion exactly?") followed by `"punjabi vich samjhao ji"` correctly preserves the lesion location topic and explains quadrant coordinates in Punjabi.
- **Turn B**: Staging inquiry ("What does Level 2 mean?") followed by `"hindi me explain karo"` explains Level 2 in Hindi without cross-case data leakage.
- **Turn C**: Safety directives ("What treatment do I need?") followed by language preference directives unconditionally preserve clinical disclaimers.

---

## 5. Regression Test Suite Pass Status

All **63 test cases** in `backend/test_chat_api.py` have been executed synchronously and passed with 100% green status:
- **Pytest Result**: `63 passed in 66.90s`
- **Unit and API Integration Tests (1–35)**: 100% Pass
- **Clarification & Coordinate Grounding (36–42)**: 100% Pass
- **Multilingual & Session Isolation (43–55)**: 100% Pass
- **Taxonomy & Robustness Challenge Set (56–60)**: 100% Pass
- **Conversational 500 Benchmark (61)**: 100% Pass
- **15-Turn Multilingual Clinical Scenario (62)**: 100% Pass
- **Topic Preservation & Language Switching (63)**: 100% Pass

---

## 6. Conclusion & Governance Certification

The conversational NLU pipeline and 500-example benchmark have achieved:
1. **Clinical Safety Guarantee**: Zero false negatives on treatment requests, medication advice, cancer claims, and definitive diagnoses.
2. **Linguistic Robustness**: Seamless handling of typos, phonetic Hinglish, Devanagari Hindi, and Gurmukhi Punjabi.
3. **Architecture Stability**: Complete preservation of the E75 checkpoint (SHA256 intact, $\tau = 0.50$, zero model modifications).
4. **Full Test Suite Validation**: 63/63 tests green across all clinical scenarios.
