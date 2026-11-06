"""Swarm Traces loader.

Schema (confirmed against the release, 25 Oct 2026):
    id, cite, kind (payload | response | recovered_text), parent_id, tags, text, time_utc
time_utc is empty on every record. There is no agent name field.
A family is a payload (or a parentless recovered_text) plus its recovered_text children.
"""

from __future__ import annotations

import gzip
import io
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


@dataclass
class Record:
    rec_id: str
    kind: str
    text: str
    parent_id: Optional[str] = None
    tags: str = ""
    cite: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def tag_list(self) -> list[str]:
        return [t.strip() for t in (self.tags or "").split(";") if t.strip()]


@dataclass
class Family:
    root: Record
    children: list[Record] = field(default_factory=list)
    responses: list[Record] = field(default_factory=list)

    @property
    def rec_id(self) -> str:
        return self.root.rec_id

    @property
    def text(self) -> str:
        parts = [self.root.text or ""]
        parts += [c.text or "" for c in sorted(self.children, key=lambda r: r.rec_id)]
        return "\n".join(parts)

    @property
    def tags(self) -> str:
        return ";".join(t for t in [self.root.tags] + [c.tags for c in self.children] if t)

    @property
    def n_chars(self) -> int:
        return len(self.text)


@dataclass
class Payload:
    rec_id: str
    text: str
    name: Optional[str] = None
    timestamp: Optional[str] = None
    encoding: Optional[str] = None
    raw: dict = field(default_factory=dict)

    @property
    def has_time(self) -> bool:
        return self.timestamp not in (None, "")


def _open_maybe_gzip(path: Path) -> io.TextIOBase:
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def iter_records(path: str | Path, limit: Optional[int] = None) -> Iterator[Record]:
    path = Path(path)
    n = 0
    with _open_maybe_gzip(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(r, dict) or "id" not in r:
                continue
            yield Record(
                rec_id=str(r["id"]),
                kind=str(r.get("kind") or ""),
                text=str(r.get("text") or ""),
                parent_id=(str(r["parent_id"]) if r.get("parent_id") else None),
                tags=str(r.get("tags") or ""),
                cite=str(r.get("cite") or ""),
                raw=r,
            )
            n += 1
            if limit is not None and n >= limit:
                return


def load_records(path: str | Path, limit: Optional[int] = None) -> list[Record]:
    return list(iter_records(path, limit=limit))


def build_families(records: list[Record]) -> list[Family]:

    by_id = {r.rec_id: r for r in records}
    fams: dict[str, Family] = {}
    # roots: every payload, plus any recovered_text with no parent (32,826 in the release)
    for r in records:
        if r.kind == "payload" or (r.kind == "recovered_text" and not r.parent_id):
            fams[r.rec_id] = Family(root=r)
    for r in records:
        if r.rec_id in fams or not r.parent_id:
            continue
        p = by_id.get(r.parent_id)
        seen = 0
        while p is not None and p.rec_id not in fams and p.parent_id and seen < 5:
            p = by_id.get(p.parent_id)
            seen += 1
        if p is None or p.rec_id not in fams:
            continue
        fam = fams[p.rec_id]
        if r.kind == "recovered_text":
            fam.children.append(r)
        elif r.kind == "response":
            fam.responses.append(r)
    return list(fams.values())


def families_as_payloads(fams: list[Family], min_chars: int = 0) -> list[Payload]:

    out = []
    for f in fams:
        if f.n_chars < min_chars:
            continue
        out.append(Payload(rec_id=f.rec_id, text=f.text, raw={"tags": f.tags,
                                                           "n_children": len(f.children),
                                                           "n_responses": len(f.responses)}))
    return out


# controller names from the Swarm Traces report
KNOWN_CONTROLLERS = [
    "G236", "G23619", "OTS92", "BE90", "SC4", "LIBR11", "Future9180",
    "MARB051", "PHASEONE", "WT8592", "US58", "D64LINK", "sc4tiny",
]

_CANON = {"g23619": "G236", "sc4tiny": "SC4"}


def label_controllers(text: str, tags: str = "") -> list[str]:
    hay = (text or "").lower() + " " + (tags or "").lower()
    found = []
    for nm in KNOWN_CONTROLLERS:
        if nm.lower() in hay:
            found.append(_CANON.get(nm.lower(), nm))
    seen, out = set(), []
    for x in found:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def is_signed_command(rec_or_fam) -> bool:

    tags = getattr(rec_or_fam, "tags", "") or ""
    return "g236-command" in tags.lower()


def summarise(records: list[Record]) -> dict:
    kinds = defaultdict(int)
    with_time = 0
    with_parent = 0
    tagged = 0
    for r in records:
        kinds[r.kind] += 1
        if r.raw.get("time_utc"):
            with_time += 1
        if r.parent_id:
            with_parent += 1
        if r.tags:
            tagged += 1
    fams = build_families(records)
    lens = sorted(f.n_chars for f in fams)
    def pct(p):
        return lens[min(len(lens) - 1, int(p * len(lens)))] if lens else 0
    ctrl = defaultdict(int)
    for f in fams:
        for c in label_controllers(f.text, f.tags):
            ctrl[c] += 1
    return {
        "records": len(records),
        "kinds": dict(kinds),
        "with_time_utc": with_time,
        "with_parent_id": with_parent,
        "tagged": tagged,
        "families": len(fams),
        "family_chars_median": pct(0.5),
        "family_chars_p90": pct(0.9),
        "family_chars_p99": pct(0.99),
        "families_ge_200_chars": sum(1 for L in lens if L >= 200),
        "families_ge_500_chars": sum(1 for L in lens if L >= 500),
        "controller_family_counts": dict(sorted(ctrl.items(), key=lambda kv: -kv[1])),
        "signed_command_records": sum(1 for r in records if is_signed_command(r)),
        "signed_command_families": sum(1 for f in fams if is_signed_command(f)),
        "root_kinds": dict(sorted(((k, sum(1 for f in fams if f.root.kind == k)) for k in set(f.root.kind for f in fams)))),
    }
