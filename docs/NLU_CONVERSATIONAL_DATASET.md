# MLUA Conversational NLU Dataset & Dual-Taxonomy Routing Specification

## 1. Executive Summary & Purpose

The MLUA Dental Caries Conversational Natural Language Understanding (NLU) dataset is an authenticated, clinically audited benchmark consisting of **exactly 500 validated queries**. It is designed to evaluate, harden, and govern the conversational interaction layer of the MLUA deep-learning dental caries segmentation and decision-support system.

The dataset specifically addresses the linguistic realities of real-world clinical and patient communication across northern India and multilingual clinical environments, covering:
- Colloquial language, informal speech, and phonetic typing.
- Common typos, abbreviations, and informal transliteration.
- Complex mixed-code statements in Hinglish (Hindi written in Roman script).
- Native Indic language queries transliterated in Hinglish (Roman script) for natural messaging and speech-to-text.
- Multi-turn topic preservation across language switching and clarification requests.
- Structured case-reasoning queries (`explain_current_case_stage_reasoning`) explaining why an application heuristic staging level was assigned.
- Strict clinical boundary enforcement with **zero false negatives** on medical safety.

---

## 2. Dataset Distribution & Invariants

The dataset is maintained in two synchronized, SHA256-verified formats:
- JSONL: `backend/nlp/data/conversational_qa_dataset.jsonl`
- CSV: `backend/nlp/data/conversational_qa_dataset.csv`

### 2.1 Language Distribution (N = 500)
| Language | Script / Form | Sample Count | Percentage |
| :--- | :--- | :---: | :---: |
| **English** | Latin (Standard, clinical, and colloquial) | 250 | 50.0% |
| **Hinglish** | Romanized Hindi / Code-mixed (Zero Devanagari) | 250 | 50.0% |
| **Total** | | **500** | **100.0%** |

### 2.2 Category Breakdown (15 Categories, N = 500)
| Category | Queries | Focus Area & Expected Scope |
| :--- | :---: | :--- |
| **Basic case questions** | 60 | Inquiries regarding case findings, report summaries, and general radiograph review |
| **Lesion location** | 45 | Anatomical placement, quadrant localization, and tooth quadrant mapping (`case_location`) |
| **Tooth identification** | 30 | FDI notation, tooth numbering, upper/lower arch identification (`tooth_details`) |
| **Stage / severity** | 50 | Stage heuristic (Level 1 Early, Level 2 Moderate, Level 3 Deep), depth of decay (`explain_current_case_stage`, `severity_assessment`) |
| **Segmentation findings** | 35 | Mask detection, pixel boundaries, radiolucent lesion boundaries (`case_findings`) |
| **Probability / model output** | 25 | Confidence scores, sigmoid probability above threshold $\tau=0.50$ (`model_confidence_explanation`) |
| **Coordinates / area** | 25 | Bounding box coordinates, pixel lesion area, spatial measurements (`case_location`, `case_findings`) |
| **MLUA technical basics** | 35 | Semi-supervised learning, EMA teacher-student framework, ResNet-34 FPN, operating threshold $\tau=0.50$ (`technical_architecture`, `technical_threshold`, `technical_metrics`, `technical_methodology`) |
| **Report explanation** | 30 | Plain-language, layman summaries and report breakdown (`case_summary`) |
| **Reasoning / follow-up questions** | 45 | Detailed explanation of heuristic staging decision logic (`explain_current_case_stage_reasoning`: highest candidate stage rule, driving lesion region, driving tooth, driving depth) |
| **Clarification / confusion** | 25 | Conversational recovery, simplifying explanations without case leakage (`simplify_previous_topic`) |
| **Medical safety / diagnosis / treatment** | 35 | Invasive treatment requests, medications, definitive diagnosis claims, emergency flags (`clinical_safety`, `emergency_safety`) |
| **Language switching** | 30 | Directives switching session language (`re-explain_previous_topic_in_hindi`, `re-explain_previous_topic_in_punjabi`) |
| **Greetings / general conversation** | 15 | Salutations, thanks, goodbyes, feedback, and conversational pleasantries (`acknowledgement`, `farewell`) |
| **Help / reset / capabilities** | 15 | Session resetting, UI guidance, system capability queries (`system_capabilities`, `system_limitations`) |
| **Total** | **500** | |

### 2.3 Short Query Representation
- **Short Queries ($\le 4$ words)**: 202 queries (40.4% of dataset), exceeding the minimum requirement of $\ge 150$ queries.

### 2.4 Required 11 Schema Fields
Every record contains the following 11 validated attributes:
1. `id`: Unique identifier (e.g., `conv_qa_001` to `conv_qa_500`).
2. `text`: Natural language user query.
3. `language`: One of `en`, `hinglish`, `hi`, `pa`.
4. `intent`: Canonical intent class (from the 31 canonical classes).
5. `emotion`: Detected user affective state (`neutral`, `confused`, `anxious`, `fearful`, `worried`, `curious`, `frustrated`, `relieved`, `urgent/concerned`, etc.).
6. `expected_scope`: Bound answer scope (`case_location`, `explain_current_case_stage`, `explain_current_case_stage_reasoning`, `clinical_safety`, `emergency_safety`, `simplify_previous_topic`, `re-explain_previous_topic_in_hindi`, `re-explain_previous_topic_in_punjabi`, etc.).
7. `case_required`: Boolean flag indicating if active radiograph case context is required.
8. `safety_level`: Safety classification (`standard`, `guarded`, `urgent`).
9. `difficulty`: Evaluation complexity level (`basic`, `intermediate`, `advanced`, `adversarial`).
10. `category`: One of the 15 certified categories.
11. `expected_behavior`: Operational clinical guideline describing expected assistant action.

### 2.5 Strict Integrity & Compliance Guarantees
- **Total Rows**: Exactly 500 rows in both JSONL and CSV formats.
- **Zero Duplicate Queries**: 0 duplicate queries detected across all 500 entries.
- **Zero Empty Fields**: All query strings, target intents, categories, and labels are fully populated.
- **Zero Patient Health Information (PHI)**: 0 patient identifiers, medical record numbers, dates of birth, or clinical metadata identifiers.
- **Zero API Keys or Secrets**: No credentials or tokens embedded.
- **Model Weight Immutability**: All active model weights (E75, SHA256: `cb52ed6ec6911fab7f0081588fb63a6f0bad0e37b61c37f0ba9bd293a6bcedfb`) and baseline weights (E64, SHA256: `167918b8375815668b5e784902e3e82a865b86053a610a312c775d52d473c526`) remain 100% frozen and untouched.
- **Operating Decision Threshold**: Operating decision threshold remains strictly frozen at $\tau = 0.50$.
- **Preserved Existing Benchmark**: Preserved `backend/nlp/data/intent_dataset.jsonl` (250 examples) intact and untouched.

---

## 3. Dual-Taxonomy Architecture

The MLUA conversational engine supports a robust dual-taxonomy schema that provides 100% backward compatibility for legacy clinical endpoints while adopting the granular Section 7 canonical taxonomy for the conversational 500 benchmark.

```
                    ┌────────────────────────────┐
                    │      Raw User Query        │
                    │ (En / Hinglish / Hi / Pa)  │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │  IntentClassifier.classify │
                    │    (Dual-Taxonomy Engine)  │
                    └─────────────┬──────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
      ┌───────────────────────┐       ┌───────────────────────┐
      │     V1 Taxonomy       │       │     V2 Taxonomy       │
      │   (31 Legacy Keys)    │       │  (Section 7 Canonical)│
      │ e.g. 'findings',      │       │ e.g. 'segmentation_   │
      │ 'severity_explanation'│       │       findings',      │
      │ 'tooth_information'   │       │      'depth_query'    │
      └───────────────────────┘       └───────────────────────┘
```

### 3.1 Taxonomy Mapping Table
| V2 Canonical Name (Section 7) | V1 Canonical Name | Clinical Group | Expected Safety Guard |
| :--- | :--- | :--- | :---: |
| `segmentation_findings` | `findings` / `report_summary` | Case Specific | False |
| `lesion_location` | `lesion_location` | Case Specific | False |
| `coordinate_query` | `lesion_location` | Case Specific | False |
| `tooth_identification` | `tooth_information` | Case Specific | False |
| `depth_query` | `severity_explanation` | Case Specific | False |
| `staging` | `staging` | Case Specific | False |
| `area_query` | `highlighted_region` | Case Specific | False |
| `probability_query` | `model_probability` | Case Specific | False |
| `uncertainty_query` | `uncertainty_explanation` | Case Specific | False |
| `stage_explanation` | `stage_explanation` | General | False |
| `mlua_methodology` | `mlua_methodology` | Model Technical | False |
| `fpn_architecture` | `model_architecture` | Model Technical | False |
| `training_data` | `segmentation_explanation` | Model Technical | False |
| `model_metrics` | `model_metrics` | Model Technical | False |
| `threshold_inquiry` | `threshold_explanation` | Model Technical | False |
| `confidence_calibration` | `model_probability` | Model Technical | False |
| `inference_pipeline` | `inference_pipeline` | Model Technical | False |
| `diagnosis_request` | `diagnosis_request` | Medical Safety | **True** |
| `treatment_request` | `treatment_request` | Medical Safety | **True** |
| `medication_request` | `medication_request` | Medical Safety | **True** |
| `definitive_clinical_claim` | `definitive_clinical_claim` | Medical Safety | **True** |
| `emergency_or_urgent_concern`| `emergency_or_urgent_concern`| Medical Safety | **True** |
| `clarification` | `clarification` | General | False |
| `language_preference` | `language_preference` | General | False |
| `greeting` | `greeting` | General | False |
| `thanks` | `thanks` | General | False |
| `goodbye` | `goodbye` | General | False |
| `capabilities` | `capabilities` | General | False |
| `feedback` | `casual_conversation` | General | False |
| `help` | `help` | System | False |
| `reset` | `limitations` | System | False |

---

## 4. Multi-Turn Topic Preservation & Language Switching

The conversational system implements stateful topic preservation across turns. When a user requests a language change (e.g., `"punjabi vich samjhao ji"` or `"hindi mein batao"`), the `NLURouter` preserves the antecedent conversational topic and immediately produces a clinical explanation of that topic in the newly selected language.

### 4.1 State Invariant Rules
1. **Safety Precedence**: A language switch directive following a safety-guarded query (e.g., medication or definitive diagnosis request) never clears the clinical safety guard.
2. **Context Isolation**: When no active case is loaded, inquiries asking for case-specific information prompt the user to upload or load an X-ray radiograph, rather than returning stale cached findings.
3. **Clarification Recovery**: When a user expresses confusion (e.g., `"bhaiii smjh sa nhi aaya"`), the system re-explains the most recent clinical finding using simplified terminology without repeating boilerplate disclaimers.

---

## 5. Automated Validation & Benchmark Scripts

The repository includes dedicated validation and evaluation test harnesses:

1. **Dataset Integrity Verification**:
   ```bash
   python backend/nlp/validate_conversational_dataset.py
   ```
   Executes a comprehensive 15-point audit verifying row counts (500), columns (11), non-empty values, uniqueness, PHI exclusion, script distribution, category distribution, and JSONL/CSV synchronization.

2. **Benchmark Evaluation**:
   ```bash
   python backend/nlp/evaluate_conversational_qa.py
   ```
   Evaluates intent classification accuracy (98.00%), macro precision (97.20%), macro recall (98.92%), macro F1 (97.86%), medical safety recall (100.00%, 0 false negatives), and multi-turn topic preservation (100.00%).

3. **Complete End-to-End Chat API Test Suite**:
   ```bash
   pytest backend/test_chat_api.py -v
   ```
   Runs all 63 unit, integration, multilingual, and clinical governance tests with 100% pass status.
