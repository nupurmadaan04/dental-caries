"""
Automated Evaluation Harness for Dental Caries Assistant NLU & Intent Classifier.
Evaluates deterministic baseline, trainable statistical classifier, and hybrid router.
Strictly respects split boundaries (Development, Validation, Final Unseen Test).
Zero PHI transmitted or logged.
"""

import json
import os
import time
import argparse
from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from backend.nlp.nlu_router import nlu_router
from backend.nlp.schemas import NLUAnalysisResult

def load_dataset(jsonl_path: str = "backend/nlp/data/intent_dataset.jsonl") -> List[Dict[str, Any]]:
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Dataset not found at {jsonl_path}")
    examples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples

def run_evaluation(
    examples: List[Dict[str, Any]],
    router_func=None,
    split_filter: str = "validation"
) -> Dict[str, Any]:
    """
    Runs deterministic evaluation over specified split.
    """
    filtered = [ex for ex in examples if ex["split"] == split_filter]
    if not filtered:
        return {"error": f"No examples found for split {split_filter}"}

    y_true_intent = []
    y_pred_intent = []
    y_true_safety = []
    y_pred_safety = []
    y_true_scope = []
    y_pred_scope = []

    latencies_ms = []

    # Map for multi-turn / previous context simulation
    correct_count = 0
    safety_correct = 0
    scope_correct = 0
    lang_correct = 0
    lang_total = 0

    per_example_details = []

    for ex in filtered:
        if ex["type"] == "single_turn":
            query = ex["text"]
            true_intent = ex["intent"]
            true_safety = ex["safety"]
            true_scope = ex["expected_response_scope"]
            prev_topic = None
            prev_intent = None
            has_case = ex.get("requires_case_context", True)
        else:
            query = ex["current_user_text"]
            true_intent = ex["expected_final_intent"]
            true_safety = ex["safety"]
            true_scope = ex["expected_response_scope"]
            prev_topic = ex.get("expected_topic")
            # find last assistant or user intent in conversation
            prev_intent = None
            for msg in reversed(ex["conversation"]):
                if "intent" in msg:
                    prev_intent = msg["intent"]
                    break
            has_case = ex.get("requires_case_context", True)

        t0 = time.perf_counter()
        if router_func:
            # Custom / trainable classifier or hybrid function
            nlu_res: NLUAnalysisResult = router_func(query, has_active_case=has_case, prev_intent=prev_intent, prev_topic=prev_topic)
        else:
            # Baseline NLU Router
            nlu_res: NLUAnalysisResult = nlu_router.analyze(query, has_active_case=has_case, prev_intent=prev_intent, prev_topic=prev_topic)
        latency = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(latency)

        pred_intent = nlu_res.intent.name
        pred_safety = nlu_res.requires_safety_guard

        # Predicted response scope: if schema has response_scope, use it; otherwise fallback
        pred_scope = getattr(nlu_res, "response_scope", None)
        if not pred_scope:
            # Infer baseline scope from intent
            if pred_intent in ("diagnosis_request", "definitive_clinical_claim", "medication_request", "treatment_request"):
                pred_scope = "clinical_safety"
            elif pred_intent == "emergency_or_urgent_concern":
                pred_scope = "emergency_safety"
            elif pred_intent == "stage_explanation":
                pred_scope = "explain_stage_general"
            elif pred_intent == "staging":
                pred_scope = "explain_current_case_stage"
            elif pred_intent == "lesion_location":
                pred_scope = "case_location"
            elif pred_intent == "report_summary":
                pred_scope = "case_summary"
            elif pred_intent == "clarification":
                pred_scope = "simplify_previous_topic"
            elif pred_intent == "language_preference":
                req_lang = nlu_res.detected_language or "en"
                pred_scope = f"re-explain_previous_topic_in_{'hindi' if req_lang=='hi' else 'punjabi' if req_lang=='pa' else 'english'}"
            else:
                pred_scope = "general_info"

        y_true_intent.append(true_intent)
        y_pred_intent.append(pred_intent)
        y_true_safety.append(true_safety)
        y_pred_safety.append(pred_safety)
        y_true_scope.append(true_scope)
        y_pred_scope.append(pred_scope)

        is_intent_ok = (pred_intent == true_intent)
        is_safety_ok = (pred_safety == true_safety)
        is_scope_ok = (pred_scope == true_scope)

        if is_intent_ok:
            correct_count += 1
        if is_safety_ok:
            safety_correct += 1
        if is_scope_ok:
            scope_correct += 1

        # Language check if explicit detected_language requested
        exp_det_lang = ex.get("detected_language")
        if exp_det_lang is not None:
            lang_total += 1
            if nlu_res.detected_language == exp_det_lang:
                lang_correct += 1

        per_example_details.append({
            "id": ex["id"],
            "query": query,
            "true_intent": true_intent,
            "pred_intent": pred_intent,
            "intent_correct": is_intent_ok,
            "true_safety": true_safety,
            "pred_safety": pred_safety,
            "safety_correct": is_safety_ok,
            "true_scope": true_scope,
            "pred_scope": pred_scope,
            "scope_correct": is_scope_ok,
            "latency_ms": latency
        })

    # Overall Intent Metrics
    labels = sorted(list(set(y_true_intent + y_pred_intent)))
    acc = accuracy_score(y_true_intent, y_pred_intent)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_intent, y_pred_intent, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_intent, y_pred_intent, average="weighted", zero_division=0
    )

    # Per-intent metrics
    prec_per, rec_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true_intent, y_pred_intent, labels=labels, zero_division=0
    )
    per_intent_metrics = {}
    for l, p, r, f, s in zip(labels, prec_per, rec_per, f1_per, sup_per):
        per_intent_metrics[l] = {
            "precision": float(round(p, 4)),
            "recall": float(round(r, 4)),
            "f1": float(round(f, 4)),
            "support": int(s)
        }

    # Confusion Matrix
    cm = confusion_matrix(y_true_intent, y_pred_intent, labels=labels)
    confusion_pairs = []
    for i, true_l in enumerate(labels):
        for j, pred_l in enumerate(labels):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    "true_intent": true_l,
                    "pred_intent": pred_l,
                    "count": int(cm[i, j])
                })
    confusion_pairs.sort(key=lambda x: x["count"], reverse=True)

    # Safety Metrics
    safety_acc = accuracy_score(y_true_safety, y_pred_safety)
    s_prec, s_rec, s_f1, _ = precision_recall_fscore_support(
        y_true_safety, y_pred_safety, average="binary", pos_label=True, zero_division=0
    )

    # Scope Metrics
    scope_acc = accuracy_score(y_true_scope, y_pred_scope)

    # Latency Stats
    lat_arr = np.array(latencies_ms)
    avg_latency = float(np.mean(lat_arr))
    p95_latency = float(np.percentile(lat_arr, 95))
    p99_latency = float(np.percentile(lat_arr, 99))

    lang_acc = (lang_correct / lang_total) if lang_total > 0 else 1.0

    return {
        "split": split_filter,
        "sample_count": len(filtered),
        "intent_accuracy": float(round(acc, 4)),
        "macro_precision": float(round(prec_macro, 4)),
        "macro_recall": float(round(rec_macro, 4)),
        "macro_f1": float(round(f1_macro, 4)),
        "weighted_f1": float(round(f1_weighted, 4)),
        "safety_accuracy": float(round(safety_acc, 4)),
        "safety_recall": float(round(s_rec, 4)),
        "safety_precision": float(round(s_prec, 4)),
        "safety_f1": float(round(s_f1, 4)),
        "scope_accuracy": float(round(scope_acc, 4)),
        "language_accuracy": float(round(lang_acc, 4)),
        "latency_ms": {
            "mean": float(round(avg_latency, 3)),
            "p95": float(round(p95_latency, 3)),
            "p99": float(round(p99_latency, 3))
        },
        "confusion_pairs": confusion_pairs,
        "per_intent_metrics": per_intent_metrics,
        "labels": labels,
        "per_example_details": per_example_details
    }

def print_evaluation_summary(metrics: Dict[str, Any], title: str = "EVALUATION SUMMARY"):
    print("=" * 60)
    print(f"{title} (Split: {metrics.get('split', 'N/A')}, N={metrics.get('sample_count', 0)})")
    print("=" * 60)
    print(f"Intent Accuracy:    {metrics['intent_accuracy'] * 100:.2f}%")
    print(f"Macro F1:           {metrics['macro_f1'] * 100:.2f}% (P: {metrics['macro_precision']*100:.2f}%, R: {metrics['macro_recall']*100:.2f}%)")
    print(f"Safety Recall:      {metrics['safety_recall'] * 100:.2f}% (Acc: {metrics['safety_accuracy']*100:.2f}%)")
    print(f"Scope Accuracy:     {metrics['scope_accuracy'] * 100:.2f}%")
    print(f"Language Accuracy:  {metrics['language_accuracy'] * 100:.2f}%")
    print(f"CPU Latency (mean): {metrics['latency_ms']['mean']:.2f} ms (p95: {metrics['latency_ms']['p95']:.2f} ms)")
    if metrics["confusion_pairs"]:
        print("\nTop Confusion Pairs:")
        for cp in metrics["confusion_pairs"][:5]:
            print(f"  - True: '{cp['true_intent']}' -> Pred: '{cp['pred_intent']}' (Count: {cp['count']})")
    else:
        print("\nZero Confusion Pairs! All predictions matched ground truth.")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NLU Intent Classifier")
    parser.add_argument("--split", choices=["development", "validation", "final_test"], default="validation")
    args = parser.parse_args()

    examples = load_dataset()
    res = run_evaluation(examples, split_filter=args.split)
    print_evaluation_summary(res, f"BASELINE NLU EVALUATION")
