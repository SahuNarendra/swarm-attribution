"""Style perturbation: rename variables, reindent, strip comments, strip controller tokens. strength 0 to 1.
"""

from __future__ import annotations

import random
import re

from .loader import Payload

_IDENT_RE = re.compile(r"\b(tmp_val_\d+|tmpVal\d+|TMP_VAL_\d+|v\d+)\b")
_COMMENT_LINE_RE = re.compile(r"(#.*?$|//.*?$)", re.M)
_CONTROLLER_RE = re.compile(r"\b([A-Z]{2,}\d{2,}|[A-Z][a-z]+\d{3,})\b")


def _rename_vars(text: str, rng: random.Random) -> str:
    mapping: dict[str, str] = {}

    def repl(m):
        k = m.group(0)
        if k not in mapping:
            mapping[k] = f"x{rng.randint(1000, 9999)}"
        return mapping[k]

    return _IDENT_RE.sub(repl, text)


def _strip_comments(text: str) -> str:
    return _COMMENT_LINE_RE.sub("", text)


def _reindent(text: str, rng: random.Random) -> str:
    target = rng.choice(["\t", "  "])
    return re.sub(r"^    ", target, text, flags=re.M)


def _strip_controllers(text: str) -> str:
    return _CONTROLLER_RE.sub("REDACTED", text)


def perturb(p: Payload, strength: float, seed: int = 0) -> Payload:
    rng = random.Random(hash((p.rec_id, seed)) & 0xFFFFFFFF)
    text = p.text or ""
    ops = []
    if strength > 0.0:
        ops.append(_rename_vars)
    if strength >= 0.25:
        ops.append(_reindent)
    if strength >= 0.5:
        ops.append(_strip_comments)
    if strength >= 0.75:
        ops.append(_strip_controllers)
    for op in ops:
        try:
            text = op(text, rng) if "rng" in op.__code__.co_varnames else op(text)
        except TypeError:
            text = op(text)
    return Payload(
        rec_id=p.rec_id + f"~p{strength:.2f}",
        text=text,
        name=p.name,
        timestamp=p.timestamp,
        encoding=("mutated" if strength >= 0.5 else p.encoding),
        raw=p.raw,
    )
