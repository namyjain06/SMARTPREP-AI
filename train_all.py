"""
SmartPrep AI - Training Pipeline
Generates all datasets and trains all models.
Run this ONCE before starting the app.
"""

import os
import sys

# Fix path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 55)
print("  SmartPrep AI — Model Training Pipeline")
print("=" * 55)

# ── 1. Generate Datasets ───────────────────────────────────
print("\n[STEP 1] Generating datasets...")
from dataset.generate_datasets import (
    generate_difficulty_dataset,
    generate_student_performance_dataset,
    generate_answer_evaluation_dataset
)

df_difficulty  = generate_difficulty_dataset("dataset/difficulty_dataset.csv")
df_students    = generate_student_performance_dataset(300, "dataset/student_performance.csv")
df_answers     = generate_answer_evaluation_dataset("dataset/answer_evaluation.csv")

# ── 2. Train Difficulty Models ─────────────────────────────
print("\n[STEP 2] Training difficulty prediction models...")
from ml_models.difficulty_predictor import train_difficulty_models

diff_results, diff_trained, diff_vec, diff_le, best_diff_name = train_difficulty_models(df_difficulty)

print("\n  📊 Model Comparison:")
print(f"  {'Model':<25} {'Accuracy':>10} {'F1-Score':>10} {'CV Mean':>10}")
print("  " + "-" * 58)
for name, m in diff_results.items():
    marker = " ⭐" if name == best_diff_name else ""
    print(f"  {name:<25} {m['accuracy']:>9}% {m['f1_score']:>9}% {m['cv_mean']:>9}%{marker}")

# ── 3. Train K-Means ──────────────────────────────────────
print("\n[STEP 3] Training K-Means clustering...")
from ml_models.weak_topic_analyzer import train_kmeans

km_result = train_kmeans(df_students)
print(f"  Cluster distribution: {dict(zip(*[range(3), [sum(km_result['labels']==c) for c in range(3)]])) }")

# ── 4. Train Answer Evaluator ─────────────────────────────
print("\n[STEP 4] Training answer evaluation model...")
from dl_models.answer_evaluator import build_answer_evaluation_model

ae_result = build_answer_evaluation_model(df_answers)

# ── 5. Summary ────────────────────────────────────────────
print("\n" + "=" * 55)
print("  ✅ All models trained and saved to /models/")
print("=" * 55)
print("\n  Files saved:")
for f in os.listdir("models"):
    print(f"    models/{f}")

print("\n  📌 Run the app with:  streamlit run app.py")
print("=" * 55)
