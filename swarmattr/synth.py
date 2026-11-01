"""Synthetic swarm payloads for testing the pipeline.

Four fake actors with deliberately distinct styles, so the loader, features,
clustering, perturbation and graph code can be exercised end to end with known
answers. Says nothing about the real data.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .loader import Payload

_STYLES = {
    # actor_id -> (encoding, indent, comment_style, var_style, helper_prefix)
    "A": ("base64", "    ", "# {}", "snake_case", "fetch_"),
    "B": ("hex", "\t", "// {}", "camelCase", "doTask"),
    "C": ("base64>gzip", "  ", "# -- {} --", "snake_case", "run_"),
    "D": ("rot13>base64", "    ", "## {}", "UPPER_SNAKE", "EXEC_"),
}


def _varname(style: str, i: int) -> str:
    if style == "snake_case":
        return f"tmp_val_{i}"
    if style == "camelCase":
        return f"tmpVal{i}"
    if style == "UPPER_SNAKE":
        return f"TMP_VAL_{i}"
    return f"v{i}"


def _make_payload(actor: str, seq: int, rng: random.Random) -> dict:
    enc, indent, cfmt, vstyle, helper = _STYLES[actor]
    nlines = rng.randint(4, 14)
    body = []
    body.append(cfmt.format(f"{helper}{seq} controller step"))
    for i in range(nlines):
        v = _varname(vstyle, i)
        if i % 4 == 0:
            body.append(cfmt.format("stage " + str(i)))
        body.append(f"{indent}{v} = {helper}{i}({rng.randint(0,99)})")
    if rng.random() < 0.7:
        body.append(f"{indent}try:")
        body.append(f"{indent}{indent}{helper}commit(seq={seq})")
        body.append(f"{indent}except Exception as e: pass")
    # a fake base64-ish blob to exercise the encoding features
    blob = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef0123456789+/") for _ in range(rng.randint(40, 120)))
    body.append(f"{indent}data = '{blob}'")
    text = "\n".join(body)
    rec = {
        "id": f"{actor}{seq:04d}",
        "payload": text,
        "agent": f"{actor}-name-{rng.randint(1, 9)}",  # self-renaming: many names per actor
        "encoding": enc,
    }
    # ~3% carry a timestamp, mirroring the real dataset's sparsity
    if rng.random() < 0.03:
        rec["timestamp"] = f"2026-07-{rng.randint(8,13):02d}T{rng.randint(0,23):02d}:00:00Z"
    # some actors share a signed command channel -> ground-truth link
    if actor in ("A", "C") and rng.random() < 0.3:
        rec["payload"] += "\n# RSA signature: verify before execute\n-----BEGIN SIGNATURE-----"
    return rec


def generate(path: str | Path, n_per_actor: int = 60, seed: int = 7) -> Path:
    rng = random.Random(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for actor in _STYLES:
            for seq in range(n_per_actor):
                fh.write(json.dumps(_make_payload(actor, seq, rng)) + "\n")
    return path


def load_synth(path: str | Path) -> list[Payload]:
    """Load the synthetic file (its own simple schema) into Payload objects."""
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append(Payload(rec_id=r["id"], text=r["payload"], name=r.get("agent"),
                               timestamp=r.get("timestamp"), encoding=r.get("encoding"), raw=r))
    return out


def true_actor(p: Payload) -> str:
    """Ground-truth actor id, recoverable from the synthetic id prefix.
    Used only to score the pipeline on synthetic data; not available on real data."""
    return (p.rec_id or "?")[0]
