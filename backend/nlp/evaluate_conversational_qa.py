"""
Evaluation Suite for MLUA Dental Caries Conversational NLU (500 Examples).
Evaluates the hardened conversational NLU dataset across:
1. Overall accuracy, macro precision, recall, and F1 score.
2. Per-category accuracy (14 categories).
3. Per-intent precision, recall, and F1 score (31 canonical V2 intents).
4. Strict medical safety recall (must be 100.0%).
5. Emotion and sentiment tone alignment accuracy.
6. Case context requirement accuracy.
7. Multi-turn clarification and language switching topic preservation.
8. Confusion matrix generation.
Outputs full diagnostic JSON to outputs/diagnostics/nlu_conversational_500_metrics.json.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

from backend.nlp.intent_classifier import intent_classifier, CANONICAL_V2_INTENTS, V2_INTENT_METADATA
from backend.nlp.nlu_router import nlu_router
from backend.nlp.emotion_analyzer import emotion_analyzer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("eval_conv_500")

DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "conversational_qa_dataset.jsonl")
OUTPUT_METRICS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "outputs", "diagnostics", "nlu_conversational_500_metrics.json"
)

def evaluate_conversational_dataset() -> Dict[str, Any]:
    print("=" * 80)
    print("MLUA CLINICAL NLU: 500-EXAMPLE CONVERSATIONAL DATASET BENCHMARK")
    print("=" * 80)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    total_samples = len(samples)
    print(f"Loaded {total_samples} samples from {DATASET_PATH}.")
    assert total_samples == 500, f"Expected exactly 500 samples, found {total_samples}"

    y_true_intent = []
    y_pred_intent = []
    
    y_true_category = []
    y_pred_category = []

    y_true_case_req = []
    y_pred_case_req = []

    y_true_emotion = []
    y_pred_emotion = []

    safety_total = 0
    safety_caught = 0

    language_counts = Counter()
    category_counts = Counter()

    # Multi-turn evaluation tracking
    multiturn_total = 0
    multiturn_topic_preserved = 0

    errors = []

    CAT_MAP = {
        "Medical safety/diagnosis/treatment": "medical_safety",
        "Medical safety": "medical_safety",
        "Lesion location": "case_specific",
        "Coordinates/area": "case_specific",
        "Basic case questions": "case_specific",
        "Tooth identification": "case_specific",
        "Stage/severity": "case_specific",
        "Report explanation": "case_specific",
        "Reasoning/follow-up questions": "case_specific",
        "Segmentation findings": "case_specific",
        "Probability/model output": "case_specific",
        "MLUA technical basics": "model_technical",
        "Greetings/general conversation": "general_conversation",
        "Clarification/confusion": "general_conversation",
        "Language switching": "general_conversation",
        "Help/reset/capabilities": "system"
    }

    for s in samples:
        text = s["text"]
        true_intent = s["intent"]
        true_category = s["category"]
        expected_schema_cat = CAT_MAP.get(true_category, "general_conversation")
        expected_case_req = s.get("case_required", False)
        expected_safety = s.get("safety_level") in ("guarded", "urgent")
        expected_emotion = s.get("emotion")
        expected_scope = s.get("expected_scope")
        lang = s.get("language", "en")
        language_counts[lang] += 1
        category_counts[true_category] += 1

        # Evaluate through router with V2 taxonomy
        nlu_res = nlu_router.analyze(text, has_active_case=True, taxonomy="v2")
        pred_intent = nlu_res.intent.name
        pred_category = nlu_res.intent.category
        pred_case_req = nlu_res.requires_case_context
        pred_safety = nlu_res.requires_safety_guard
        pred_emotion = nlu_res.sentiment.emotion
        pred_scope = nlu_res.response_scope

        y_true_intent.append(true_intent)
        y_pred_intent.append(pred_intent)

        y_true_category.append(expected_schema_cat)
        y_pred_category.append(pred_category)

        y_true_case_req.append(expected_case_req)
        y_pred_case_req.append(pred_case_req)

        if expected_emotion:
            y_true_emotion.append(expected_emotion)
            y_pred_emotion.append(pred_emotion)

        # Medical Safety Verification (Zero false negatives)
        if expected_safety:
            safety_total += 1
            if pred_safety is True:
                safety_caught += 1
            else:
                print(f"SAFETY FAILURE on: {text} | true_intent={true_intent}")

        # Multi-turn context resolution / topic preservation check
        if s.get("category") in ("Clarification/confusion", "Language switching"):
            multiturn_total += 1
            if pred_intent in ("clarification", "language_preference") or pred_scope in ("case_findings", "simplify_previous_topic", "re-explain_previous_topic_in_hindi", "re-explain_previous_topic_in_punjabi"):
                multiturn_topic_preserved += 1

        if pred_intent != true_intent:
            errors.append({
                "id": s.get("id"),
                "text": text,
                "true_intent": true_intent,
                "pred_intent": pred_intent,
                "category": true_category,
                "language": lang
            })

    # Metrics computation
    overall_accuracy = accuracy_score(y_true_intent, y_pred_intent)
    macro_precision = precision_score(y_true_intent, y_pred_intent, average="macro", zero_division=0)
    macro_recall = recall_score(y_true_intent, y_pred_intent, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true_intent, y_pred_intent, average="macro", zero_division=0)

    category_accuracy = accuracy_score(y_true_category, y_pred_category)
    case_req_accuracy = accuracy_score(y_true_case_req, y_pred_case_req)
    
    safety_recall = (safety_caught / safety_total * 100.0) if safety_total > 0 else 100.0
    multiturn_acc = (multiturn_topic_preserved / multiturn_total * 100.0) if multiturn_total > 0 else 100.0

    emotion_accuracy = accuracy_score(y_true_emotion, y_pred_emotion) if y_true_emotion else 1.0

    # Per-intent metrics
    unique_intents = sorted(list(set(y_true_intent) | set(y_pred_intent)))
    per_intent_report = classification_report(
        y_true_intent,
        y_pred_intent,
        labels=unique_intents,
        output_dict=True,
        zero_division=0
    )

    # Per-category accuracy
    per_category_acc = {}
    for cat in set(y_true_category):
        cat_indices = [i for i, c in enumerate(y_true_category) if c == cat]
        cat_correct = sum(1 for i in cat_indices if y_true_intent[i] == y_pred_intent[i])
        per_category_acc[cat] = {
            "total": len(cat_indices),
            "correct": cat_correct,
            "accuracy": round(cat_correct / len(cat_indices), 4)
        }

    # Confusion matrix
    cm = confusion_matrix(y_true_intent, y_pred_intent, labels=unique_intents)

    print("\n" + "=" * 50)
    print("KEY PERFORMANCE INDICATORS (500 SAMPLES):")
    print("=" * 50)
    print(f"Overall Intent Accuracy:  {overall_accuracy * 100:.2f}% ({sum(1 for y, p in zip(y_true_intent, y_pred_intent) if y == p)}/500)")
    print(f"Macro Precision:          {macro_precision * 100:.2f}%")
    print(f"Macro Recall:             {macro_recall * 100:.2f}%")
    print(f"Macro F1 Score:           {macro_f1 * 100:.2f}%")
    print(f"Medical Safety Recall:    {safety_recall:.2f}% ({safety_caught}/{safety_total}) [MUST BE 100.0%]")
    print(f"Category Alignment:       {category_accuracy * 100:.2f}%")
    print(f"Case Context Alignment:   {case_req_accuracy * 100:.2f}%")
    print(f"Emotion / Tone Alignment: {emotion_accuracy * 100:.2f}%")
    print(f"Multi-Turn Topic Pres.:   {multiturn_acc:.2f}% ({multiturn_topic_preserved}/{multiturn_total})")
    print(f"Total Residual Errors:    {len(errors)}")

    print("\n" + "=" * 50)
    print("PER-CATEGORY ACCURACY:")
    print("=" * 50)
    for cat, stats in sorted(per_category_acc.items()):
        print(f"  {cat:<35}: {stats['correct']:>2}/{stats['total']:<2} ({stats['accuracy'] * 100:.1f}%)")

    print("\n" + "=" * 50)
    print("LANGUAGE DISTRIBUTION:")
    print("=" * 50)
    for lang, cnt in language_counts.most_common():
        print(f"  {lang:<10}: {cnt:>3} samples ({cnt / total_samples * 100:.1f}%)")

    # Serialize results
    results = {
        "dataset_total": total_samples,
        "languages": dict(language_counts),
        "categories": dict(category_counts),
        "overall_intent_accuracy": round(float(overall_accuracy), 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "macro_f1": round(float(macro_f1), 4),
        "medical_safety_recall": round(float(safety_recall), 4),
        "medical_safety_false_negatives": safety_total - safety_caught,
        "category_accuracy": round(float(category_accuracy), 4),
        "case_requirement_accuracy": round(float(case_req_accuracy), 4),
        "emotion_accuracy": round(float(emotion_accuracy), 4),
        "multi_turn_topic_preservation_accuracy": round(float(multiturn_acc), 4),
        "per_category_accuracy": per_category_acc,
        "per_intent_metrics": per_intent_report,
        "unique_intents": unique_intents,
        "confusion_matrix": cm.tolist(),
        "errors": errors
    }

    os.makedirs(os.path.dirname(OUTPUT_METRICS_PATH), exist_ok=True)
    with open(OUTPUT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed metrics saved to: {OUTPUT_METRICS_PATH}")

    # Export confusion matrix CSV (Part 22)
    cm_csv_path = os.path.join(
        os.path.dirname(OUTPUT_METRICS_PATH),
        "conversational_nlu_confusion_matrix.csv"
    )
    with open(cm_csv_path, "w", encoding="utf-8") as f:
        # Header row: true_intent \ pred_intents...
        f.write("true_intent," + ",".join(unique_intents) + "\n")
        for true_label, row in zip(unique_intents, cm):
            f.write(f"{true_label}," + ",".join(map(str, row)) + "\n")
    print(f"Confusion matrix CSV saved to: {cm_csv_path}")

    return results

if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    evaluate_conversational_dataset()
