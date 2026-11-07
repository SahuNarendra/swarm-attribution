"""summarise PATH             corpus stats
feasibility PATH           per controller feature means and separation ratio
signed PATH --out FILE     write the g236-command records to FILE, in tag order
attribute PATH             attribute families to lineages with refusal; anchors from --anchors or controller names
selftest                   synthetic end to end run
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from . import __version__
from .loader import (load_records, build_families, label_controllers, summarise, is_signed_command, Payload)
from .features import (feature_matrix, structural_features, FEATURE_NAMES, CLUSTER_FEATURES,
                       work_signature, work_signature_matrix)
from .cluster import cluster, cluster_repeated, score_against_truth
from .adversary import perturb
from .graph import build_graph, containment_order, actor_projection
from .attribute import Attributor
from . import synth


def _print(d):
    print(json.dumps(d, indent=2, ensure_ascii=False))


def cmd_summarise(args):
    recs = load_records(args.path, limit=args.limit)
    _print(summarise(recs))


def cmd_feasibility(args):
    import numpy as np
    recs = load_records(args.path, limit=args.limit)
    fams = build_families(recs)
    min_chars = args.min_chars
    groups: dict[str, list] = {}
    for f in fams:
        if f.n_chars < min_chars:
            continue
        labels = label_controllers(f.text, f.tags)
        if len(labels) != 1:
            continue
        groups.setdefault(labels[0], []).append(
            structural_features(Payload(rec_id=f.rec_id, text=f.text)))
    groups = {k: v for k, v in groups.items() if len(v) >= args.min_n}
    if not groups:
        print("No controller groups met the thresholds."); return
    report = {}
    for tag, rows in groups.items():
        M = np.array([[r[n] for n in FEATURE_NAMES] for r in rows], dtype=float)
        report[tag] = {"n": len(rows),
                       "mean": {n: float(M[:, i].mean()) for i, n in enumerate(FEATURE_NAMES)},
                       "std": {n: float(M[:, i].std()) for i, n in enumerate(FEATURE_NAMES)}}
    sep = {}
    for i, n in enumerate(FEATURE_NAMES):
        means = np.array([report[t]["mean"][n] for t in report])
        stds = np.array([report[t]["std"][n] for t in report])
        within = float(stds.mean()) + 1e-9
        sep[n] = round(float(means.std()) / within, 3)
    top = sorted(sep.items(), key=lambda kv: -kv[1])
    print(f"Feasibility check: {len(groups)} controller groups, families >= {min_chars} chars, n >= {args.min_n}")
    print("groups:", {t: report[t]["n"] for t in report})
    print("\nSeparation ratio per feature (spread of group means / mean within-group std).")
    for n, s in top:
        print(f"  {s:6.2f}  {n}")
    print("\nTop 6 features, mean by controller:")
    for n, _ in top[:6]:
        print(f"  {n}:")
        for t in report:
            print(f"      {t:<12} mean {report[t]['mean'][n]:10.2f}  std {report[t]['std'][n]:8.2f}  (n={report[t]['n']})")
    strong = sum(1 for _, s in top if s >= 1.0)
    print(f"\nFeatures with separation >= 1.0: {strong} of {len(FEATURE_NAMES)}")


def cmd_signed(args):
    recs = [r for r in load_records(args.path) if is_signed_command(r)]
    recs.sort(key=lambda r: (r.tags, r.rec_id))
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps({"id": r.rec_id, "kind": r.kind, "parent_id": r.parent_id,
                                 "tags": r.tags, "chars": len(r.text), "text": r.text}) + "\n")
    print(len(recs), "records ->", args.out)
    for r in recs:
        print(f"  {r.rec_id}  {r.kind:<15} {len(r.text):>6}  {r.tags}")


def cmd_attribute(args):
    """Anchors: a JSON lines file of {"id": record id, "lineage": name}, or, without --anchors, every family
    that names exactly one known controller. Every other family of --min-chars or more is scored.
    Output: one JSON line per family: id, lineage (null = refused), score, runner-up."""
    from collections import Counter
    recs = load_records(args.path, limit=args.limit)
    fams = [f for f in build_families(recs) if f.n_chars >= args.min_chars]
    anchor_of = {}
    if args.anchors:
        for line in open(args.anchors, encoding="utf-8"):
            if line.strip():
                d = json.loads(line); anchor_of[d["id"]] = d["lineage"]
    else:
        for f in fams:
            labs = label_controllers(f.text, f.tags)
            if len(labs) == 1: anchor_of[f.rec_id] = labs[0]
    cnt = Counter(anchor_of.values())
    keep = {c for c, n in cnt.items() if n >= args.min_anchors}
    anchors = [f for f in fams if anchor_of.get(f.rec_id) in keep]
    rest = [f for f in fams if f.rec_id not in anchor_of]
    print(f"anchors: {len(anchors)} families, {len(keep)} lineages {dict(sorted((c, cnt[c]) for c in keep))}; scoring {len(rest)} families; threshold {args.threshold}", file=sys.stderr)
    att = Attributor(threshold=args.threshold).fit([Payload(rec_id=f.rec_id, text=f.text) for f in anchors], [anchor_of[f.rec_id] for f in anchors])
    verdicts = att.attribute([Payload(rec_id=f.rec_id, text=f.text) for f in rest])
    out = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    for v in verdicts:
        out.write(json.dumps({"id": v.rec_id, "lineage": v.lineage, "score": round(v.score, 3), "runner_up": v.second, "runner_up_score": round(v.second_score, 3)}) + "\n")
    if args.out: out.close()
    got = Counter(v.lineage for v in verdicts)
    print(f"attributed {sum(n for c, n in got.items() if c)} of {len(verdicts)} ({sum(n for c, n in got.items() if c) / max(1, len(verdicts)):.2%}); refused {got.get(None, 0)}; by lineage {dict(sorted((c, n) for c, n in got.items() if c))}", file=sys.stderr)


def cmd_selftest(args):
    print(f"swarmattr {__version__} selftest\n")
    with tempfile.TemporaryDirectory() as td:
        path = synth.generate(Path(td) / "synth.jsonl", n_per_actor=args.n)
        payloads = synth.load_synth(path)
        print("1. loaded", len(payloads), "synthetic payloads")
        X, names = feature_matrix(payloads, CLUSTER_FEATURES)
        print(f"2. feature matrix: {X.shape[0]} x {X.shape[1]}")
        res = cluster(X)
        print(f"3. clustering: {res.method} inferred {res.n_clusters} ({res.notes})")
        truth = [synth.true_actor(p) for p in payloads]
        _print(score_against_truth(res.labels, truth))
        rep = cluster_repeated(X, runs=10)
        print("3b. repeated:", rep.summary())
        W, vec = work_signature_matrix(payloads, min_df=1)
        print(f"3c. work signature: {W.shape[0]} x {W.shape[1]}, e.g. {work_signature(payloads[0])[:60]!r}")
        p0 = payloads[0]
        for s in (0.0, 0.5, 1.0):
            print(f"4. perturb {s:.2f}: len {len(p0.text)} -> {len(perturb(p0, s).text)}")
        actor_of = {p.rec_id: synth.true_actor(p) for p in payloads}
        tr, te = payloads[0::2], payloads[1::2]
        att = Attributor(threshold=0.5).fit(tr, [actor_of[p.rec_id] for p in tr])
        vs = att.attribute(te)
        ok = sum(1 for v in vs if v.lineage == actor_of[v.rec_id]); ref = sum(1 for v in vs if v.refused)
        print(f"4b. attribute with refusal: {ok} of {len(vs)} right, {ref} refused")
        g = build_graph(payloads, actor_of)
        ap = actor_projection(g)
        print(f"5. graph {g.number_of_nodes()} nodes {g.number_of_edges()} edges; actor projection {ap.number_of_edges()} edges")
        print("   cut first:", containment_order(g)[:3])
    print("\nok")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="swarmattr", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarise"); s.add_argument("path"); s.add_argument("--limit", type=int)
    s.set_defaults(func=cmd_summarise)
    f = sub.add_parser("feasibility"); f.add_argument("path"); f.add_argument("--limit", type=int)
    f.add_argument("--min-chars", type=int, default=200); f.add_argument("--min-n", type=int, default=5)
    f.set_defaults(func=cmd_feasibility)
    g = sub.add_parser("signed"); g.add_argument("path"); g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_signed)
    a = sub.add_parser("attribute"); a.add_argument("path"); a.add_argument("--anchors"); a.add_argument("--out")
    a.add_argument("--threshold", type=float, default=0.66); a.add_argument("--min-chars", type=int, default=200)
    a.add_argument("--min-anchors", type=int, default=8); a.add_argument("--limit", type=int)
    a.set_defaults(func=cmd_attribute)
    t = sub.add_parser("selftest"); t.add_argument("--n", type=int, default=60)
    t.set_defaults(func=cmd_selftest)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
