"""Style perturbation: rename identifiers, reindent, strip comments, strip controller tokens. strength 0 to 1.
"""

from __future__ import annotations

import random
import re

from .loader import Payload, KNOWN_CONTROLLERS

_IDENT_RE = re.compile(r"(?<![\w])([A-Za-z_][A-Za-z0-9_]{2,})(?![\w])")
_URL_PATH_RE = re.compile(r"(https?://\S+|(?<![\w])/[\w./-]+)")
_COMMENT_LINE_RE = re.compile(r"(^[ \t]*(#|//).*?$\n?|[ \t]+(#|//).*?$)", re.M)
_CONTROLLER_RE = re.compile(r"\b(" + "|".join(sorted(KNOWN_CONTROLLERS, key=len, reverse=True)) + r")\b", re.I)
_INDENT_RE = re.compile(r"^(\t+| +)", re.M)

_KEEP = set("""
if else elif for while do break continue return def function func class struct import from as in is not and or
try except catch finally raise throw with lambda yield pass None True False null true false var let const new
this self int str float bool list dict set tuple print len range open echo exit cat curl wget bash sh python
python3 sudo chmod export http https www com org net bin usr etc tmp dev null main args argv env
Exception Error ValueError TypeError KeyError IOError OSError object string number
""".lower().split())
_KEEP |= {c.lower() for c in KNOWN_CONTROLLERS}


def _rename_idents(text: str, rng: random.Random) -> str:
    mapping: dict[str, str] = {}
    saved: list[str] = []

    def mask(m):
        saved.append(m.group(0))
        return f"\x00{len(saved) - 1}\x00"

    def repl(m):
        k = m.group(1)
        if k.lower() in _KEEP or (k.isupper() and len(k) <= 3):
            return k
        if k not in mapping:
            mapping[k] = "v" + "".join(rng.choice("abcdefghjkmnpqrstuvwxyz") for _ in range(rng.randint(3, 7)))
        return mapping[k]

    text = _URL_PATH_RE.sub(mask, text)
    text = _IDENT_RE.sub(repl, text)
    return re.sub(r"\x00(\d+)\x00", lambda m: saved[int(m.group(1))], text)


def _strip_comments(text: str) -> str:
    return _COMMENT_LINE_RE.sub("", text)


def _reindent(text: str, rng: random.Random) -> str:
    unit = rng.choice(["\t", "  ", "    "])

    def repl(m):
        s = m.group(1)
        depth = len(s) if s.startswith("\t") else max(1, len(s) // 2)
        return unit * depth

    return _INDENT_RE.sub(repl, text)


def _strip_controllers(text: str) -> str:
    return _CONTROLLER_RE.sub("X", text)


def perturb(p: Payload, strength: float, seed: int = 0) -> Payload:
    rng = random.Random(f"{p.rec_id}:{seed}")
    text = p.text or ""
    if strength > 0.0:
        text = _rename_idents(text, rng)
    if strength >= 0.25:
        text = _reindent(text, rng)
    if strength >= 0.5:
        text = _strip_comments(text)
    if strength >= 0.75:
        text = _strip_controllers(text)
    return Payload(rec_id=p.rec_id + f"~p{strength:.2f}", text=text, raw=p.raw)
