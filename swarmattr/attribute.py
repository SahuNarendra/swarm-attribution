"""Attribution with refusal. Train on anchored artefacts, score the rest, refuse below a threshold."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .loader import Payload, KNOWN_CONTROLLERS
from .features import feature_matrix, CLUSTER_FEATURES, work_signature

_CTRL_RE = re.compile("|".join(sorted(map(re.escape, KNOWN_CONTROLLERS), key=len, reverse=True)), re.I)


def strip_names(text: str) -> str:
    return _CTRL_RE.sub("", text or "")


@dataclass
class Verdict:
    rec_id: str
    lineage: str | None
    score: float
    second: str | None
    second_score: float

    @property
    def refused(self) -> bool:
        return self.lineage is None


class Attributor:
    """Two lenses (style random forest, work signature logistic regression), averaged.
    Threshold 0.66 is the setting at which 5 percent of files from lineages never seen in training were
    wrongly attributed on the July 2026 incident data (analysis/s5_anchored.py)."""

    def __init__(self, threshold: float = 0.66, seed: int = 0, min_df: int = 2):
        self.threshold = threshold
        self.seed = seed
        self.min_df = min_df

    def _feats(self, payloads):
        pl = [Payload(rec_id=p.rec_id, text=strip_names(p.text)) for p in payloads]
        X, _ = feature_matrix(pl, CLUSTER_FEATURES)
        return X, [work_signature(p) for p in pl]

    def fit(self, anchors: list[Payload], lineages: list[str]) -> "Attributor":
        if len(set(lineages)) < 2:
            raise ValueError("need anchors from at least two lineages")
        X, W = self._feats(anchors)
        y = np.array(lineages)
        self.scaler = StandardScaler().fit(X)
        self.rf = RandomForestClassifier(300, random_state=self.seed, class_weight="balanced", n_jobs=2).fit(self.scaler.transform(X), y)
        self.tv = TfidfVectorizer(token_pattern=r"\S+", min_df=min(self.min_df, max(1, len(anchors) // 10)), sublinear_tf=True).fit(W)
        self.lr = LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced").fit(self.tv.transform(W), y)
        self.classes = self.rf.classes_
        return self

    def proba(self, payloads: list[Payload]) -> np.ndarray:
        X, W = self._feats(payloads)
        return (self.rf.predict_proba(self.scaler.transform(X)) + self.lr.predict_proba(self.tv.transform(W))) / 2

    def attribute(self, payloads: list[Payload]) -> list[Verdict]:
        P = self.proba(payloads)
        out = []
        for p, row in zip(payloads, P):
            order = np.argsort(-row)
            top, sec = order[0], order[1] if len(order) > 1 else order[0]
            lineage = str(self.classes[top]) if row[top] >= self.threshold else None
            out.append(Verdict(p.rec_id, lineage, float(row[top]), str(self.classes[sec]), float(row[sec])))
        return out
