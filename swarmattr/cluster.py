"""Clustering without a fixed K. HDBSCAN by default, agglomerative with silhouette sweep for small inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClusterResult:
    labels: list[int]
    n_clusters: int
    method: str
    notes: str = ""


@dataclass
class RepeatResult:
    n_clusters: list[int] = field(default_factory=list)
    noise_frac: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        import numpy as np
        a = np.asarray(self.n_clusters, dtype=float)
        return {
            "runs": len(a),
            "n_clusters_mean": float(a.mean()) if len(a) else 0.0,
            "n_clusters_std": float(a.std()) if len(a) else 0.0,
            "n_clusters_min": int(a.min()) if len(a) else 0,
            "n_clusters_max": int(a.max()) if len(a) else 0,
            "noise_frac_mean": float(np.mean(self.noise_frac)) if self.noise_frac else 0.0,
        }


def _scale(X, min_std: float = 1e-3):
    # drop near-constant columns first; a few outliers in a dead column otherwise dominate distances
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    X = np.asarray(X, dtype=float)
    keep = X.std(axis=0) > min_std
    return StandardScaler().fit_transform(X[:, keep])


def cluster(X, method: str = "hdbscan", min_cluster_size: int = 5, min_samples: int | None = None) -> ClusterResult:
    import numpy as np
    Xs = _scale(X)

    if method == "hdbscan":
        from sklearn.cluster import HDBSCAN
        cl = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = cl.fit_predict(Xs)
        n = len(set(labels.tolist()) - {-1})
        noise = int((labels == -1).sum())
        return ClusterResult(labels.tolist(), n, "hdbscan", f"noise={noise}")

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
    return ClusterResult(lab.tolist(), k, "agglomerative", f"silhouette={score:.3f}")


def cluster_repeated(X, runs: int = 20, frac: float = 0.8, seed: int = 0, **kw) -> RepeatResult:
    import numpy as np
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    m = max(2, int(n * frac))
    out = RepeatResult()
    for _ in range(runs):
        idx = rng.choice(n, size=m, replace=False)
        r = cluster(X[idx], **kw)
        out.n_clusters.append(r.n_clusters)
        out.noise_frac.append(sum(1 for l in r.labels if l == -1) / m)
    return out


def score_against_truth(labels: list[int], truth: list[str]) -> dict:
    from sklearn.metrics import (adjusted_rand_score, homogeneity_score,
                                 completeness_score)
    return {
        "n_inferred": len(set(l for l in labels if l != -1)),
        "n_true": len(set(truth)),
        "adjusted_rand": round(adjusted_rand_score(truth, labels), 3),
        "homogeneity": round(homogeneity_score(truth, labels), 3),
        "completeness": round(completeness_score(truth, labels), 3),
    }
