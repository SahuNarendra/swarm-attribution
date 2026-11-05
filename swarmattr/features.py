"""Per payload features. Two lenses:
1. structural: how the text is written (23 features, 20 for clustering)
2. work signature: what it reaches for (command heads, imports, helper names), as a TF-IDF document
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

from .loader import Payload, KNOWN_CONTROLLERS

_B64_RE = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
_HEX_RE = re.compile(r"\b(?:0x)?[0-9a-fA-F]{32,}\b")
_URL_RE = re.compile(r"https?://[^\s'\"]+")
_COMMENT_RE = re.compile(r"(#.*?$|//.*?$|/\*.*?\*/)", re.M | re.S)
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FUNC_DEF_RE = re.compile(r"\b(?:def|function|func)\s+([A-Za-z_][A-Za-z0-9_]*)")
_CONTROLLER_RE = re.compile(r"\b(" + "|".join(sorted(KNOWN_CONTROLLERS, key=len, reverse=True)) + r")\b", re.I)
_TRY_RE = re.compile(r"\b(try|except|catch|finally|rescue)\b")
_SEG_SPLIT_RE = re.compile(r"[;&|\n]+")
_HEAD_RE = re.compile(r"[A-Za-z0-9_./\-]+")
_IMPORT_RE = re.compile(r"^\s*(?:from\s+([A-Za-z_][\w.]*)\s+import|import\s+([A-Za-z_][\w.]*))", re.M)
_SNAKE_RE = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b")
_CAMEL_RE = re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]+)+\b")


def _shannon(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def structural_features(p: Payload) -> dict[str, float]:
    t = p.text or ""
    n_chars = len(t)
    lines = t.splitlines() or [""]
    n_lines = len(lines)
    idents = _IDENT_RE.findall(t)
    comments = _COMMENT_RE.findall(t)
    comment_text = " ".join(m if isinstance(m, str) else m[0] for m in comments)
    snake = _SNAKE_RE.findall(t)
    camel = _CAMEL_RE.findall(t)

    feats = {
        "len_chars": float(n_chars),
        "n_lines": float(len(lines)),
        "mean_line_len": _safe_div(n_chars, len(lines)),
        "max_line_len": float(max((len(l) for l in lines), default=0)),
        "entropy": _shannon(t),

        "b64_per_line": _safe_div(len(_B64_RE.findall(t)), n_lines),
        "hex_per_line": _safe_div(len(_HEX_RE.findall(t)), n_lines),
        "urls_per_line": _safe_div(len(_URL_RE.findall(t)), n_lines),

        "idents_per_line": _safe_div(len(idents), n_lines),
        "unique_ident_ratio": _safe_div(len(set(idents)), len(idents)),
        "mean_ident_len": _safe_div(sum(len(i) for i in idents), len(idents)),
        "snake_ratio": _safe_div(len(snake), len(snake) + len(camel)),
        "n_controller_tokens": float(len(_CONTROLLER_RE.findall(t))),

        "comment_ratio": _safe_div(len(comment_text), n_chars),
        "comments_per_line": _safe_div(len(comments), n_lines),
        "mean_comment_len": _safe_div(len(comment_text), max(1, len(comments))),

        "try_per_line": _safe_div(len(_TRY_RE.findall(t)), n_lines),
        "tab_line_ratio": _safe_div(len(re.findall(r"^\t", t, re.M)), n_lines),
        "indent4_line_ratio": _safe_div(len(re.findall(r"^    ", t, re.M)), n_lines),
        "indent2_line_ratio": _safe_div(len(re.findall(r"^  (?! )", t, re.M)), n_lines),
        "semicolon_ratio": _safe_div(t.count(";"), n_lines),
        "blank_line_ratio": _safe_div(sum(1 for l in lines if not l.strip()), n_lines),
        "trailing_ws_ratio": _safe_div(sum(1 for l in lines if l != l.rstrip()), n_lines),
    }
    return feats


def helper_fingerprints(p: Payload, top: int = 0) -> list[str]:
    names = [m.lower() for m in _FUNC_DEF_RE.findall(p.text or "")]
    return names if top == 0 else names[:top]


def controller_tokens(p: Payload) -> list[str]:
    return sorted(set(m.upper() for m in _CONTROLLER_RE.findall(p.text or "")))


FEATURE_NAMES = sorted(structural_features(Payload(rec_id="_", text="x=1")).keys())

# for clustering: no size features, no controller names
CLUSTER_FEATURES = [n for n in FEATURE_NAMES
                    if n not in ("len_chars", "n_lines", "n_controller_tokens")]


def feature_matrix(payloads: list[Payload], names: Optional[list[str]] = None):
    import numpy as np
    names = list(names or FEATURE_NAMES)
    rows = []
    for p in payloads:
        f = structural_features(p)
        rows.append([f[n] for n in names])
    return np.asarray(rows, dtype=float), names


def command_heads(text: str) -> list[str]:
    heads = []
    for seg in _SEG_SPLIT_RE.split(text or ""):
        seg = seg.strip()
        m = _HEAD_RE.match(seg)
        if m and not re.match(r"\s*=(?!=)", seg[m.end():]):  # skip assignments
            heads.append(m.group(0).split("/")[-1].lower())
    return heads


def work_signature(p: Payload) -> str:
    t = p.text or ""
    toks = ["cmd:" + h for h in command_heads(t)]
    toks += ["imp:" + (a or b).split(".")[0].lower() for a, b in _IMPORT_RE.findall(t)]
    toks += ["fn:" + n for n in helper_fingerprints(p)]
    return " ".join(toks)


def work_signature_matrix(payloads: list[Payload], min_df: int = 3, vectorizer=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [work_signature(p) for p in payloads]
    if vectorizer is None:
        vectorizer = TfidfVectorizer(token_pattern=r"\S+", min_df=min_df, sublinear_tf=True)
        X = vectorizer.fit_transform(docs)
    else:
        X = vectorizer.transform(docs)
    return X, vectorizer
