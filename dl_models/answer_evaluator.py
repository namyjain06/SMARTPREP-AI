"""
SmartPrep AI - Answer Evaluation using NLP (TF-IDF + Cosine Similarity)
Evaluates student answers against reference answers.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.text_processor import (
    clean_text, compute_cosine_similarity, score_from_similarity, prepare_answer_eval_features
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def build_answer_evaluation_model(df: pd.DataFrame):
    """
    Build answer evaluation pipeline.
    Uses TF-IDF + cosine similarity as primary metric.
    Trains a Ridge regression to map similarity → score.
    """
    # Compute similarities
    df_eval = prepare_answer_eval_features(df)
    
    X = df_eval[["cosine_similarity"]].values
    y = df_eval["score"].values.astype(float)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    regressor = Ridge(alpha=1.0)
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    y_pred_clipped = np.clip(y_pred, 0, 5)
    
    mse = mean_squared_error(y_test, y_pred_clipped)
    r2 = r2_score(y_test, y_pred_clipped)
    mae = np.mean(np.abs(y_test - y_pred_clipped))
    
    # Save model
    with open(os.path.join(MODELS_DIR, "answer_eval_regressor.pkl"), "wb") as f:
        pickle.dump(regressor, f)
    
    print(f"  [✓] Answer Evaluator trained | MSE: {mse:.4f} | R²: {r2:.4f} | MAE: {mae:.4f}")
    
    return {
        "model": regressor,
        "mse": round(mse, 4),
        "r2": round(r2, 4),
        "mae": round(mae, 4),
        "df_eval": df_eval
    }


def evaluate_answer(student_answer: str, reference_answer: str, max_score: int = 5,
                    regressor=None) -> dict:
    """
    Evaluate a student's answer against the reference.
    Returns score, similarity, and detailed feedback.
    """
    sim = compute_cosine_similarity(reference_answer, student_answer)
    
    # Predict score using regressor or fallback to similarity-based scoring
    if regressor is not None:
        try:
            predicted_score = float(np.clip(regressor.predict([[sim]])[0], 0, max_score))
        except Exception:
            predicted_score = score_from_similarity(sim, max_score)
    else:
        predicted_score = score_from_similarity(sim, max_score)
    
    predicted_score = round(predicted_score, 2)
    
    # Generate feedback
    feedback = _generate_feedback(sim, predicted_score, max_score, student_answer, reference_answer)
    
    # Grade
    pct = predicted_score / max_score
    if pct >= 0.8:
        grade = "Excellent ✅"
    elif pct >= 0.6:
        grade = "Good 🟢"
    elif pct >= 0.4:
        grade = "Average 🟡"
    else:
        grade = "Needs Improvement 🔴"
    
    return {
        "student_answer": student_answer,
        "reference_answer": reference_answer,
        "cosine_similarity": round(sim * 100, 2),
        "predicted_score": predicted_score,
        "max_score": max_score,
        "grade": grade,
        "percentage": round(pct * 100, 2),
        "feedback": feedback
    }


def _generate_feedback(sim: float, score: float, max_score: int,
                       student: str, reference: str) -> list:
    """Generate actionable feedback based on similarity and content analysis."""
    feedback = []
    
    student_words = set(clean_text(student).split())
    ref_words = set(clean_text(reference).split())
    
    # Remove stopwords
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "it", "in", "of", "and", "or", "to"}
    student_kw = student_words - stopwords
    ref_kw = ref_words - stopwords
    
    overlap = student_kw & ref_kw
    missing = ref_kw - student_kw
    
    overlap_pct = len(overlap) / max(len(ref_kw), 1)
    
    if sim >= 0.8:
        feedback.append("✅ Excellent coverage of key concepts.")
    elif sim >= 0.6:
        feedback.append("🟢 Good answer with most key points covered.")
    elif sim >= 0.4:
        feedback.append("🟡 Partial answer — some important concepts missing.")
    else:
        feedback.append("🔴 Answer needs significant improvement.")
    
    if len(student.split()) < 10:
        feedback.append("⚠️ Answer is too brief. Elaborate with examples and explanations.")
    
    if missing:
        top_missing = list(missing)[:5]
        feedback.append(f"📌 Missing key terms: {', '.join(top_missing)}")
    
    if overlap_pct > 0.6:
        feedback.append(f"✔ Good keyword coverage ({int(overlap_pct*100)}% of key terms matched).")
    
    return feedback


def load_answer_eval_model():
    """Load answer evaluation regressor from disk."""
    try:
        with open(os.path.join(MODELS_DIR, "answer_eval_regressor.pkl"), "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
