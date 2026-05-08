"""
SmartPrep AI - Unsupervised Learning: K-Means Clustering for Weak Topic Identification
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

SUBJECTS = ["Mathematics", "Physics", "Computer Science", "Chemistry", "Biology"]
N_CLUSTERS = 3  # Weak / Average / Strong


def train_kmeans(df: pd.DataFrame):
    """Train K-Means on student subject scores."""
    X = df[SUBJECTS].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train K-Means
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_scaled)
    
    labels = kmeans.labels_
    sil_score = silhouette_score(X_scaled, labels)
    
    # Interpret clusters: rank by mean score
    cluster_means = []
    for c in range(N_CLUSTERS):
        mask = labels == c
        mean_score = df[SUBJECTS][mask].mean().mean()
        cluster_means.append((c, mean_score))
    
    sorted_clusters = sorted(cluster_means, key=lambda x: x[1])
    cluster_map = {
        sorted_clusters[0][0]: "Weak",
        sorted_clusters[1][0]: "Average",
        sorted_clusters[2][0]: "Strong"
    }
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Save models
    with open(os.path.join(MODELS_DIR, "kmeans_model.pkl"), "wb") as f:
        pickle.dump(kmeans, f)
    with open(os.path.join(MODELS_DIR, "kmeans_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_DIR, "kmeans_pca.pkl"), "wb") as f:
        pickle.dump(pca, f)
    
    print(f"  [✓] K-Means trained | Silhouette Score: {sil_score:.4f}")
    
    return {
        "model": kmeans,
        "scaler": scaler,
        "pca": pca,
        "labels": labels,
        "cluster_map": cluster_map,
        "silhouette_score": round(sil_score, 4),
        "X_pca": X_pca,
        "cluster_means": {c: round(m, 2) for c, m in cluster_means}
    }


def identify_weak_topics_for_student(scores: dict, kmeans, scaler, cluster_map) -> dict:
    """
    Given a student's subject scores, identify weak topics.
    scores: {"Mathematics": 45, "Physics": 72, ...}
    """
    score_vector = np.array([scores.get(s, 50) for s in SUBJECTS]).reshape(1, -1)
    X_scaled = scaler.transform(score_vector)
    
    cluster = kmeans.predict(X_scaled)[0]
    performance_group = cluster_map.get(cluster, "Average")
    
    # Identify weak subjects (below 50)
    weak = [s for s in SUBJECTS if scores.get(s, 50) < 50]
    moderate = [s for s in SUBJECTS if 50 <= scores.get(s, 50) < 65]
    strong = [s for s in SUBJECTS if scores.get(s, 50) >= 65]
    
    # Compute subject-wise status
    subject_status = {}
    for s in SUBJECTS:
        score = scores.get(s, 50)
        if score < 50:
            subject_status[s] = {"score": score, "status": "🔴 Weak", "priority": "High"}
        elif score < 65:
            subject_status[s] = {"score": score, "status": "🟡 Moderate", "priority": "Medium"}
        else:
            subject_status[s] = {"score": score, "status": "🟢 Strong", "priority": "Low"}
    
    recommendations = []
    for s in weak:
        recommendations.append(f"Dedicate 3+ hours/day to {s} — scores critically low ({scores[s]}%)")
    for s in moderate:
        recommendations.append(f"Spend 1.5-2 hours/day on {s} — needs improvement ({scores[s]}%)")
    
    return {
        "performance_group": performance_group,
        "cluster_id": int(cluster),
        "weak_subjects": weak,
        "moderate_subjects": moderate,
        "strong_subjects": strong,
        "subject_status": subject_status,
        "recommendations": recommendations
    }


def get_class_weak_topic_analysis(df: pd.DataFrame, labels: np.ndarray, cluster_map: dict):
    """Aggregate weak topic analysis for the entire class."""
    df = df.copy()
    df["cluster"] = labels
    df["performance"] = df["cluster"].map(cluster_map)
    
    analysis = {}
    for subj in SUBJECTS:
        analysis[subj] = {
            "mean_score": round(df[subj].mean(), 2),
            "std_score": round(df[subj].std(), 2),
            "min_score": round(df[subj].min(), 2),
            "max_score": round(df[subj].max(), 2),
            "pct_weak": round((df[subj] < 50).mean() * 100, 2),
            "pct_strong": round((df[subj] >= 65).mean() * 100, 2),
        }
    
    return analysis


def load_kmeans_models():
    """Load K-Means models from disk."""
    try:
        with open(os.path.join(MODELS_DIR, "kmeans_model.pkl"), "rb") as f:
            kmeans = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "kmeans_scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "kmeans_pca.pkl"), "rb") as f:
            pca = pickle.load(f)
        return kmeans, scaler, pca
    except FileNotFoundError:
        return None, None, None
