"""Structural and stylistic features per payload. Describes how the text is written, not what it does.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

from .loader import Payload

_B64_RE = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
_HEX_RE = re.compile(r"\b(?:0x)?[0-9a-fA-F]{32,}\b")
_URL_RE = re.compile(r"https?://[^\s'\"]+")
_COMMENT_RE = re.compile(r"(#.*?$|//.*?$|/\*.*?\*/)", re.M | re.S)
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FUNC_DEF_RE = re.compile(r"\b(?:def|function|func)\s+([A-Za-z_][A-Za-z0-9_]*)")
_CONTROLLER_RE = re.compile(r"\b([A-Z]{2,}\d{2,}|[A-Z][a-z]+\d{3,})\b")
_CONTROLLER_NOISE = {"SHA256", "SHA512", "SHA1", "RS256", "HS256", "ES256", "ED25519",
                     "AES256", "AES128", "UTF8", "MD5", "RSA2048", "RSA4096", "X509"}
_TRY_RE = re.compile(r"\b(try|except|catch|finally|rescue)\b")
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

        "n_b64_blobs": float(len(_B64_RE.findall(t))),
        "n_hex_blobs": float(len(_HEX_RE.findall(t))),
        "n_urls": float(len(_URL_RE.findall(t))),
        "has_encoding_field": 1.0 if p.encoding else 0.0,

        "n_idents": float(len(idents)),
        "unique_ident_ratio": _safe_div(len(set(idents)), len(idents)),
        "mean_ident_len": _safe_div(sum(len(i) for i in idents), len(idents)),
        "snake_ratio": _safe_div(len(snake), len(snake) + len(camel)),
        "n_controller_tokens": float(sum(1 for m in _CONTROLLER_RE.findall(t)
                                         if m.upper() not in _CONTROLLER_NOISE)),

        "comment_ratio": _safe_div(len(comment_text), n_chars),
        "n_comments": float(len(comments)),
        "mean_comment_len": _safe_div(len(comment_text), max(1, len(comments))),

        "n_try": float(len(_TRY_RE.findall(t))),
        "indent_tabs": float(t.count("\t")),
        "indent_spaces_4": float(len(re.findall(r"^    ", t, re.M))),
        "indent_spaces_2": float(len(re.findall(r"^  (?! )", t, re.M))),
        "semicolon_ratio": _safe_div(t.count(";"), max(1, len(lines))),
    }
    return feats


def helper_fingerprints(p: Payload, top: int = 0) -> list[str]:
    names = [m.lower() for m in _FUNC_DEF_RE.findall(p.text or "")]
    return names if top == 0 else names[:top]


def controller_tokens(p: Payload) -> list[str]:
    return sorted(set(m for m in _CONTROLLER_RE.findall(p.text or "")
                      if m.upper() not in _CONTROLLER_NOISE))


FEATURE_NAMES = sorted(structural_features(Payload(rec_id="_", text="x=1")).keys())


def feature_matrix(payloads: list[Payload]):
    import numpy as np
    rows = []
    for p in payloads:
        f = structural_features(p)
        rows.append([f[name] for name in FEATURE_NAMES])
    return np.asarray(rows, dtype=float), list(FEATURE_NAMES)
