"""
Trainable Local Intent Classifier for Dental Caries Assistant NLU.
Trained strictly on the Development split (70%, 175 examples).
Tuned and evaluated strictly on the Validation split (15%, 38 examples).
Final unseen test split (15%, 37 examples) remains strictly UNTOUCHED.
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def load_split_data(split_name: str, jsonl_path: str = "backend/nlp/data/intent_dataset.jsonl") -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    texts = []
    labels = []
    meta = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            if item["split"] == split_name:
                t = item["text"] if item["type"] == "single_turn" else item["current_user_text"]
                l = item["intent"] if item["type"] == "single_turn" else item["expected_final_intent"]
                texts.append(t)
                labels.append(l)
                meta.append(item)
    return texts, labels, meta

def train_and_evaluate_classifier():
    X_train, y_train, train_meta = load_split_data("development")
    X_val, y_val, val_meta = load_split_data("validation")

    print(f"Loaded {len(X_train)} training examples from development split.")
    print(f"Loaded {len(X_val)} validation examples from validation split.")

    # Pipeline with character and word n-grams to handle multilingual scripts (Devanagari, Gurmukhi), Hinglish, typos
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            lowercase=True,
            sublinear_tf=True,
            min_df=1
        )),
        ("clf", LogisticRegression(
            C=5.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])

    pipeline.fit(X_train, y_train)

    train_preds = pipeline.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    train_f1 = precision_recall_fscore_support(y_train, train_preds, average="macro", zero_division=0)[2]

    val_preds = pipeline.predict(X_val)
    val_probs = pipeline.predict_proba(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = precision_recall_fscore_support(y_val, val_preds, average="macro", zero_division=0)[2]

    print("\n" + "=" * 50)
    print("TRAINABLE CLASSIFIER (LOGISTIC REGRESSION) METRICS")
    print("=" * 50)
    print(f"Train Accuracy: {train_acc * 100:.2f}%, Train Macro F1: {train_f1 * 100:.2f}%")
    print(f"Val Accuracy:   {val_acc * 100:.2f}%, Val Macro F1:   {val_f1 * 100:.2f}%")
    print("=" * 50)

    # Save trained model artifact for router integration
    model_dir = os.path.join("backend", "nlp", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "local_intent_classifier.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Saved trained classifier to {model_path}")

    return pipeline

if __name__ == "__main__":
    train_and_evaluate_classifier()
