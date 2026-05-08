"""
SmartPrep AI - Preprocessing Pipeline
Text cleaning, TF-IDF vectorization, and similarity computation.
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, remove special chars, extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_texts(texts):
    return [clean_text(t) for t in texts]


# ─────────────────────────────────────────────
# TF-IDF VECTORIZER
# ─────────────────────────────────────────────

def build_tfidf_vectorizer(texts, max_features=5000, ngram_range=(1, 2)):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        sublinear_tf=True
    )
    cleaned = preprocess_texts(texts)
    X = vectorizer.fit_transform(cleaned)
    return vectorizer, X


# ─────────────────────────────────────────────
# ANSWER SIMILARITY
# ─────────────────────────────────────────────

def compute_cosine_similarity(reference: str, student_answer: str, vectorizer=None) -> float:
    """Compute cosine similarity between reference and student answer."""
    ref_clean = clean_text(reference)
    stu_clean = clean_text(student_answer)
    
    if vectorizer is None:
        local_vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        try:
            matrix = local_vec.fit_transform([ref_clean, stu_clean])
        except Exception:
            return 0.0
    else:
        try:
            matrix = vectorizer.transform([ref_clean, stu_clean])
        except Exception:
            return 0.0
    
    sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return float(sim)


def score_from_similarity(similarity: float, max_score: int = 5) -> float:
    """Convert similarity score [0,1] to grade [0, max_score]."""
    score = round(similarity * max_score, 2)
    return min(score, max_score)


# ─────────────────────────────────────────────
# FEATURE ENGINEERING FOR DIFFICULTY PREDICTION
# ─────────────────────────────────────────────

def prepare_difficulty_features(df: pd.DataFrame):
    """Return X (tfidf + metadata), y (labels), vectorizer, encoder."""
    texts = df["question"].tolist()
    cleaned = preprocess_texts(texts)
    
    # TF-IDF on question text
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english")
    X_text = vectorizer.fit_transform(cleaned)
    
    # Subject encoding
    le = LabelEncoder()
    subject_enc = le.fit_transform(df["subject"].values).reshape(-1, 1)
    
    # Metadata features: word count, char count
    word_counts = df["question"].apply(lambda x: len(str(x).split())).values.reshape(-1, 1)
    char_counts = df["question"].apply(lambda x: len(str(x))).values.reshape(-1, 1)
    
    from scipy.sparse import hstack, csr_matrix
    X_meta = csr_matrix(np.hstack([subject_enc, word_counts, char_counts]))
    X = hstack([X_text, X_meta])
    
    y = df["difficulty_label"].values
    return X, y, vectorizer, le


def prepare_student_performance_features(df: pd.DataFrame):
    """Return subject score matrix for clustering."""
    subject_cols = ["Mathematics", "Physics", "Computer Science", "Chemistry", "Biology"]
    X = df[subject_cols].values
    return X, subject_cols


# ─────────────────────────────────────────────
# ANSWER EVALUATION PREPROCESSING
# ─────────────────────────────────────────────

def prepare_answer_eval_features(df: pd.DataFrame):
    """Compute similarity features for answer evaluation dataset."""
    sims = []
    for _, row in df.iterrows():
        sim = compute_cosine_similarity(row["reference_answer"], row["student_answer"])
        sims.append(sim)
    df = df.copy()
    df["cosine_similarity"] = sims
    df["predicted_score"] = df["cosine_similarity"].apply(lambda s: score_from_similarity(s, max_score=5))
    return df
