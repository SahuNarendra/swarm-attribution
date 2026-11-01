"""Clustering and actor-count harness.

Infers the number of actors from the feature matrix rather than taking a fixed K.
On synthetic data the result can be scored against the known actor labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ClusterResult:
    labels: list[int]
    n_clusters: int
    method: str
    notes: str = ""


def cluster(X, method: str = "auto", min_cluster_size: int = 5) -> ClusterResult:
    """Cluster feature rows into inferred actors.

    method="auto": use HDBSCAN if installed (infers K, marks noise as -1),
    else fall back to a simple K-free agglomerative sweep picking the K with the
    best silhouette. Both are standard; the choice is recorded for the report.
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)

    if method in ("auto", "hdbscan"):
        try:
            import hdbscan  # type: ignore
            cl = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            labels = cl.fit_predict(Xs)
            n = len(set(labels) - {-1})
            return ClusterResult(labels.tolist(), n, "hdbscan",
                                 f"noise={int((labels == -1).sum())}")
        except Exception:
            if method == "hdbscan":
                raise
            # fall through to agglomerative

    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    best = None
    n_samples = Xs.shape[0]
    hi = min(20, max(2, n_samples - 1))
    for k in range(2, hi + 1):
        try:
            lab = AgglomerativeClustering(n_clusters=k).fit_predict(Xs)
            score = silhouette_score(Xs, lab)
        except Exception:
            continue
        if best is None or score > best[0]:
            best = (score, k, lab)
    if best is None:
        return ClusterResult([0] * n_samples, 1, "fallback", "degenerate")
    score, k, lab = best
    return ClusterResult(lab.tolist(), k, "agglomerative",
                         f"silhouette={score:.3f}")


def score_against_truth(labels: list[int], truth: list[str]) -> dict:
    """Synthetic-only: how well do inferred clusters match known actors?
    Uses adjusted Rand index and homogeneity. Not valid on real data (no truth)."""
    from sklearn.metrics import (adjusted_rand_score, homogeneity_score,
                                 completeness_score)
    return {
        "n_inferred": len(set(l for l in labels if l != -1)),
        "n_true": len(set(truth)),
        "adjusted_rand": round(adjusted_rand_score(truth, labels), 3),
        "homogeneity": round(homogeneity_score(truth, labels), 3),
        "completeness": round(completeness_score(truth, labels), 3),
    }
