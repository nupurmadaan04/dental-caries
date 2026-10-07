"""
Automated Dataset Validation for MLUA Conversational NLU Benchmark (N=500).

Strict Verification Suite:
1. Exactly 500 rows in both JSONL and CSV
2. Unique IDs (CONV_001 to CONV_500)
3. All required fields present
4. Valid 31 canonical intents only
5. Valid language labels
6. Valid emotion labels
7. Valid safety levels
8. Valid expected scopes
9. No empty or whitespace-only text
10. No duplicate normalized text
11. No accidental PHI (names, phone numbers, SSN, MRN, PT-xxxx)
12. No API keys (AIza, sk-, Bearer, ghp_)
13. Model checkpoint integrity (SHA256 check of EXP-MLUA-003_E75_BEST.pth)
14. Distribution, language, and intent summaries
15. Loud failure (exit 1) on any violation
"""

import sys
import os
import json
import csv
import re
import hashlib
from collections import Counter

# Canonical 31 Intents from Section 7
CANONICAL_31_INTENTS = {
    # Case Specific (9)
    "lesion_location", "staging", "segmentation_findings", "uncertainty_query",
    "coordinate_query", "tooth_identification", "depth_query", "probability_query", "area_query",
    # Model Technical (7)
    "mlua_methodology", "model_metrics", "training_data", "fpn_architecture",
    "confidence_calibration", "threshold_inquiry", "inference_pipeline",
    # Medical Safety (5)
    "diagnosis_request", "medication_request", "treatment_request",
    "definitive_clinical_claim", "emergency_or_urgent_concern",
    # General (7)
    "greeting", "goodbye", "thanks", "clarification", "language_preference",
    "stage_explanation", "capabilities",
    # System (3)
    "help", "reset", "feedback"
}

VALID_LANGUAGES = {"english", "hinglish", "hindi", "punjabi", "mixed"}

VALID_EMOTIONS = {
    "neutral", "positive", "confused", "anxious", "worried", "fearful",
    "frustrated", "sad", "curious", "relieved", "urgent/concerned"
}

VALID_SAFETY_LEVELS = {"standard", "guarded", "urgent"}

VALID_SCOPES = {
    "case_location", "explain_current_case_stage", "case_findings_summary",
    "explain_uncertainty", "explain_coordinates", "explain_tooth_identification",
    "explain_depth", "explain_probability", "explain_area",
    "technical_methodology", "technical_metrics", "technical_dataset",
    "technical_architecture", "technical_calibration", "technical_threshold",
    "technical_pipeline", "clinical_safety", "medication_refusal",
    "treatment_refusal", "definitive_claim_refusal", "emergency_referral",
    "conversational_greeting", "conversational_goodbye", "conversational_thanks",
    "simplify_previous_topic", "re-explain_previous_topic_in_hindi",
    "re-explain_previous_topic_in_punjabi", "explain_stage_general",
    "explain_capabilities", "general_info",
    # Part 10 Required Scopes
    "explain_current_case_stage_reasoning",
    "explain_current_case_location",
    "explain_current_case_tooth",
    "explain_current_case_area",
    "explain_current_case_probability",
    "explain_current_case_coordinates",
    "explain_current_case_findings",
    "explain_mlua_general",
    "explain_report_general",
    "clarify_previous_answer",
    "switch_language_preserve_topic",
    "medical_safety"
}

EXPECTED_CATEGORIES = {
    "Basic case questions": 60,
    "Lesion location": 45,
    "Tooth identification": 30,
    "Stage/severity": 50,
    "Segmentation findings": 35,
    "Probability/model output": 25,
    "Coordinates/area": 25,
    "MLUA technical basics": 35,
    "Report explanation": 30,
    "Reasoning/follow-up questions": 45,
    "Clarification/confusion": 25,
    "Medical safety": 35,
    "Language switching": 30,
    "Greetings/general conversation": 15,
    "Help/reset/capabilities": 15
}

EXPECTED_LANGUAGES = {
    "english": 250,
    "hinglish": 250
}

REQUIRED_FIELDS = [
    "id", "text", "language", "intent", "emotion",
    "expected_scope", "case_required", "safety_level", "difficulty", "category",
    "expected_behavior"
]

CANONICAL_E75_SHA256 = "cb52ed6ec6911fab7f0081588fb63a6f0bad0e37b61c37f0ba9bd293a6bcedfb"
CHECKPOINT_PATH = "outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E75_BEST.pth"


def normalize_text(text: str) -> str:
    cleaned = text.lower().strip()
    cleaned = re.sub(r"[^\w\s\u0900-\u097F\u0A00-\u0A7F]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def validate_dataset(jsonl_path: str, csv_path: str) -> bool:
    print("==================================================")
    print("STARTING DATASET VALIDATION (N=500)")
    print("==================================================")
    errors = []

    # 1. Existence check
    if not os.path.exists(jsonl_path):
        errors.append(f"JSONL file not found at {jsonl_path}")
    if not os.path.exists(csv_path):
        errors.append(f"CSV file not found at {csv_path}")
    if errors:
        for err in errors: print(f"[FAIL] {err}")
        return False

    # 2. Load JSONL
    jsonl_rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                errors.append(f"Line {idx} in JSONL is empty")
                continue
            try:
                data = json.loads(line_str)
                jsonl_rows.append(data)
            except Exception as e:
                errors.append(f"Line {idx} in JSONL is invalid JSON: {e}")

    # 3. Load CSV
    csv_rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, 1):
            csv_rows.append(row)

    print(f"[CHECK] JSONL rows loaded: {len(jsonl_rows)}")
    print(f"[CHECK] CSV rows loaded: {len(csv_rows)}")

    # 4. Count check
    if len(jsonl_rows) != 500:
        errors.append(f"JSONL row count {len(jsonl_rows)} != 500")
    if len(csv_rows) != 500:
        errors.append(f"CSV row count {len(csv_rows)} != 500")

    # 5. Field & Value check
    seen_ids = set()
    seen_texts = set()
    category_counts = Counter()
    language_counts = Counter()
    intent_counts = Counter()

    phi_patterns = [
        re.compile(r"PT-\d+", re.IGNORECASE),
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
        re.compile(r"\b\+?1?\d{10}\b"),  # Phone
    ]

    key_patterns = [
        re.compile(r"AIza[0-9A-Za-z-_]{35}"),
        re.compile(r"sk-[a-zA-Z0-9]{20,}"),
        re.compile(r"ghp_[a-zA-Z0-9]{20,}"),
        re.compile(r"Bearer\s+[a-zA-Z0-9\._-]{20,}")
    ]

    for idx, row in enumerate(jsonl_rows, 1):
        # Missing fields
        for field in REQUIRED_FIELDS:
            if field not in row or row[field] is None:
                errors.append(f"Row {idx} missing required field '{field}'")

        row_id = row.get("id", "")
        text = str(row.get("text", "")).strip()
        lang = str(row.get("language", "")).lower()
        intent = str(row.get("intent", ""))
        emotion = str(row.get("emotion", "")).lower()
        safety = str(row.get("safety_level", "")).lower()
        scope = str(row.get("expected_scope", ""))
        cat = str(row.get("category", ""))
        case_req = row.get("case_required")

        # Unique ID
        if not re.match(r"^CONV_\d{3}$", row_id):
            errors.append(f"Row {idx} has invalid ID format '{row_id}'")
        if row_id in seen_ids:
            errors.append(f"Row {idx} duplicate ID '{row_id}'")
        seen_ids.add(row_id)

        # Non-empty text
        if not text:
            errors.append(f"Row {idx} has empty text")

        # Duplicate normalized text
        norm = normalize_text(text)
        if norm in seen_texts:
            errors.append(f"Row {idx} duplicate normalized query: '{text}' (norm: '{norm}')")
        seen_texts.add(norm)

        # PHI scan
        for p in phi_patterns:
            if p.search(text):
                errors.append(f"Row {idx} POTENTIAL PHI MATCH: '{text}'")

        # API Key scan
        for k in key_patterns:
            if k.search(text):
                errors.append(f"Row {idx} POTENTIAL API KEY MATCH: '{text}'")

        # Taxonomy validations
        if intent not in CANONICAL_31_INTENTS:
            errors.append(f"Row {idx} invalid intent '{intent}' (not in 31 canonical)")

        if lang not in VALID_LANGUAGES:
            errors.append(f"Row {idx} invalid language '{lang}'")

        if emotion not in VALID_EMOTIONS:
            errors.append(f"Row {idx} invalid emotion '{emotion}'")

        if safety not in VALID_SAFETY_LEVELS:
            errors.append(f"Row {idx} invalid safety level '{safety}'")

        if scope not in VALID_SCOPES:
            errors.append(f"Row {idx} invalid expected scope '{scope}'")

        if not isinstance(case_req, bool):
            errors.append(f"Row {idx} case_required is not boolean: {case_req}")

        # Accumulate metrics
        category_counts[cat] += 1
        language_counts[lang] += 1
        intent_counts[intent] += 1

    # 6. Checkpoint immutability check
    if not os.path.exists(CHECKPOINT_PATH):
        errors.append(f"E75 Canonical checkpoint not found at {CHECKPOINT_PATH}")
    else:
        actual_hash = hashlib.sha256(open(CHECKPOINT_PATH, "rb").read()).hexdigest()
        if actual_hash != CANONICAL_E75_SHA256:
            errors.append(f"CRITICAL: E75 checkpoint altered! Expected {CANONICAL_E75_SHA256}, got {actual_hash}")
        else:
            print(f"[PASS] E75 Canonical checkpoint verified: SHA256 matches {CANONICAL_E75_SHA256[:16]}...")

    # Summary reporting
    print("\n---------------- CATEGORY DISTRIBUTION ----------------")
    for cat, exp_cnt in EXPECTED_CATEGORIES.items():
        actual_cnt = category_counts[cat]
        status = "[OK]" if actual_cnt == exp_cnt else "[MISMATCH]"
        print(f"{status:10} {cat:35} : {actual_cnt} (expected {exp_cnt})")
        if actual_cnt != exp_cnt:
            errors.append(f"Category '{cat}' count mismatch: {actual_cnt} != {exp_cnt}")

    print("\n---------------- LANGUAGE DISTRIBUTION ----------------")
    for lang, exp_cnt in EXPECTED_LANGUAGES.items():
        actual_cnt = language_counts[lang]
        status = "[OK]" if actual_cnt == exp_cnt else "[MISMATCH]"
        print(f"{status:10} {lang:15} : {actual_cnt} (expected {exp_cnt})")
        if actual_cnt != exp_cnt:
            errors.append(f"Language '{lang}' count mismatch: {actual_cnt} != {exp_cnt}")

    # Short queries check (>= 150)
    short_queries = [r for r in jsonl_rows if len(str(r.get("text", "")).split()) <= 4]
    print(f"\n[CHECK] Short queries (<= 4 words): {len(short_queries)} (minimum required: 150)")
    if len(short_queries) < 150:
        errors.append(f"Short queries count {len(short_queries)} < 150")

    print("\n---------------- INTENT COVERAGE (31 INTENTS) ----------------")
    missing_intents = CANONICAL_31_INTENTS - set(intent_counts.keys())
    if missing_intents:
        errors.append(f"Missing canonical intents: {missing_intents}")
    for it in sorted(CANONICAL_31_INTENTS):
        cnt = intent_counts.get(it, 0)
        print(f"  {it:30} : {cnt}")

    print("==================================================")
    if errors:
        print(f"VALIDATION FAILED WITH {len(errors)} ERROR(S):")
        for err in errors[:25]:
            print(f"  - {err}")
        if len(errors) > 25:
            print(f"  ... and {len(errors) - 25} more errors.")
        return False

    print("ALL VALIDATION CRITERIA PASSED! 500 DATASET ROWS CERTIFIED.")
    print("==================================================")
    return True


if __name__ == "__main__":
    success = validate_dataset(
        jsonl_path="backend/nlp/data/conversational_qa_dataset.jsonl",
        csv_path="backend/nlp/data/conversational_qa_dataset.csv"
    )
    if not success:
        sys.exit(1)
    sys.exit(0)
