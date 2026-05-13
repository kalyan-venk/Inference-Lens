"""
Unsupervised clustering for response quality archetype discovery.

Three clustering approaches are applied to sentence embedding vectors:
  K-Means          sweep K from 2 to 12, select via elbow and silhouette
  DBSCAN           density-based, surfaces noise points and irregular shapes
  Hierarchical     Ward linkage with dendrogram for visual merge structure

Cluster quality is evaluated with:
  Silhouette score
  Davies-Bouldin index
  Elbow plot (inertia vs K for K-Means)

After optimal K is selected, clusters are interpreted by:
  Inspecting top examples per cluster
  Cross-tabulating cluster membership with preference labels
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_kmeans(
    embeddings: np.ndarray,
    k_range: range = range(2, 13),
    random_state: int = 42,
    n_init: int = 10,
) -> dict:
    """Fit K-Means for each K in k_range. Returns results dict."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    results = {}
    for k in k_range:
        logger.info("Fitting K-Means with k=%d", k)
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(embeddings)
        results[k] = {
            "labels": labels,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(embeddings, labels, sample_size=5000, random_state=random_state),
            "davies_bouldin": davies_bouldin_score(embeddings, labels),
        }
        logger.info(
            "k=%d  inertia=%.2f  silhouette=%.4f  DB=%.4f",
            k, results[k]["inertia"], results[k]["silhouette"], results[k]["davies_bouldin"],
        )
    return results


def run_dbscan(
    embeddings: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 10,
) -> np.ndarray:
    """Fit DBSCAN. Label -1 indicates noise points."""
    from sklearn.cluster import DBSCAN

    logger.info("Fitting DBSCAN (eps=%.2f, min_samples=%d)", eps, min_samples)
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = model.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    logger.info("DBSCAN found %d clusters, %d noise points", n_clusters, n_noise)
    return labels


def run_hierarchical(
    embeddings: np.ndarray,
    n_clusters: int = 5,
    linkage: str = "ward",
) -> np.ndarray:
    """Fit agglomerative hierarchical clustering."""
    from sklearn.cluster import AgglomerativeClustering

    logger.info("Fitting hierarchical clustering (n_clusters=%d, linkage=%s)", n_clusters, linkage)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(embeddings)
    return labels


def summarize_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    text_col: str = "chosen",
    label_col: str = "preference_label",
    top_n: int = 3,
) -> pd.DataFrame:
    """Build a cluster summary table with size, label distribution, and example texts."""
    df = df.copy()
    df["cluster"] = labels

    rows = []
    for cluster_id in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cluster_id]
        row = {
            "cluster": cluster_id,
            "size": len(subset),
            "pct_chosen": subset[label_col].mean() if label_col in subset.columns else None,
            "examples": subset[text_col].iloc[:top_n].tolist(),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    return summary
