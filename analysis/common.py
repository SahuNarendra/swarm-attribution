"""Shared pieces for the analysis scripts. Run every script from the repo root:
python analysis/<script>.py
Outputs go to analysis/out/ (ignored by git).
"""
import math, os, pickle, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
DATA = os.path.join(ROOT, "data", "redacted.jsonl.gz")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)

from swarmattr.loader import KNOWN_CONTROLLERS  # noqa: E402

CTRL_RE = re.compile("|".join(sorted(map(re.escape, KNOWN_CONTROLLERS), key=len, reverse=True)), re.I)


def strip_names(text):
    return CTRL_RE.sub("", text or "")


def ptype(t):
    h = t[:400].lower()
    if re.search(r"^\s*(#!/bin/|curl |wget |bash |sh |echo |export )", t, re.M): return "shell"
    if re.search(r"\b(import |def |print\(|python)\b", h): return "python"
    if re.search(r"<html|<div|<script", h): return "html"
    if re.search(r"^\s*[\{\[]", t): return "json"
    if re.search(r"\b(select |insert |from )\b", h): return "sql"
    return "other"


def out(name):
    return os.path.join(OUT, name)


def load(name):
    return pickle.load(open(out(name), "rb"))


def save(obj, name):
    pickle.dump(obj, open(out(name), "wb"))


def threshold_at(scores, y, fpr):
    neg = np.sort(scores[~y])
    return neg[int(math.ceil((1 - fpr) * len(neg))) - 1]
