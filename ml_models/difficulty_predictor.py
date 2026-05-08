"""
SmartPrep AI - Supervised Learning Models for Difficulty Prediction
Trains and evaluates Logistic Regression, Random Forest, SVM.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.text_processor import prepare_difficulty_features

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

LABEL_NAMES = ["Easy", "Medium", "Hard"]


def train_difficulty_models(df: pd.DataFrame):
    """Train all three classifiers and return metrics."""
    X, y, vectorizer, label_encoder = prepare_difficulty_features(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        
        results[name] = {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "cv_mean": round(cv_scores.mean() * 100, 2),
            "cv_std": round(cv_scores.std() * 100, 2),
            "report": classification_report(y_test, y_pred, target_names=LABEL_NAMES),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
        trained_models[name] = model
        print(f"    Accuracy: {acc*100:.2f}%")
    
    # Save best model
    best_model_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = trained_models[best_model_name]
    
    with open(os.path.join(MODELS_DIR, "difficulty_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    with open(os.path.join(MODELS_DIR, "difficulty_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODELS_DIR, "difficulty_label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    
    print(f"\n  [✓] Best model: {best_model_name} ({results[best_model_name]['accuracy']}% accuracy)")
    
    return results, trained_models, vectorizer, label_encoder, best_model_name


def predict_difficulty(question: str, subject: str, model, vectorizer, label_encoder) -> dict:
    """Predict difficulty of a single question."""
    from preprocessing.text_processor import clean_text
    from scipy.sparse import hstack, csr_matrix
    
    cleaned = clean_text(question)
    X_text = vectorizer.transform([cleaned])
    
    try:
        subj_enc = label_encoder.transform([subject])[0]
    except Exception:
        subj_enc = 0
    
    word_count = len(question.split())
    char_count = len(question)
    X_meta = csr_matrix([[subj_enc, word_count, char_count]])
    X = hstack([X_text, X_meta])
    
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
    
    label = LABEL_NAMES[pred]
    confidence = round(float(max(proba)) * 100, 2) if proba is not None else None
    
    return {
        "question": question,
        "subject": subject,
        "predicted_difficulty": label,
        "confidence": confidence,
        "probabilities": {LABEL_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(proba)} if proba is not None else {}
    }


def load_difficulty_model():
    """Load trained model from disk."""
    try:
        with open(os.path.join(MODELS_DIR, "difficulty_model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "difficulty_vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "difficulty_label_encoder.pkl"), "rb") as f:
            label_encoder = pickle.load(f)
        return model, vectorizer, label_encoder
    except FileNotFoundError:
        return None, None, None
