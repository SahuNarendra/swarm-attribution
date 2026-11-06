from common import *
import networkx as nx
from collections import defaultdict, Counter
d = load("prep.pkl"); rows = d["rows"]
a = load("arts.pkl"); arts, df = a["arts"], a["df"]
o = load("s5_out.pkl"); lab, s, T = o["lab"], o["s"], o["T"]
for name, actor in [("labelled families only", {i: r["label"] for i, r in enumerate(rows) if r["label"]}),
                    ("labelled + corpus attributions @10%", {**{i: lab[i] for i in range(len(rows)) if s[i] >= T[0.10]}, **{i: r["label"] for i, r in enumerate(rows) if r["label"]}})]:
    touched = defaultdict(Counter)
    for i, c in actor.items():
        for x in arts[i]: touched[x][c] += 1
    g = nx.Graph(); ctr = sorted(set(actor.values())); g.add_nodes_from(("A", c) for c in ctr)
    shared = {x: cs for x, cs in touched.items() if len(cs) >= 2}
    common = {x for x in shared if df[x] > 500}
    for x, cs in shared.items():
        if x in common: continue
        for c, w in cs.items(): g.add_edge(("A", c), ("X", x), weight=w)
    act = [("A", c) for c in ctr]
    def ncomp(h): return len({frozenset(nx.node_connected_component(h, n)) for n in act})
    base = ncomp(g)
    rowsout = []
    for n in [n for n in g if n[0] == "X"]:
        h = g.copy(); h.remove_node(n); rowsout.append((n[1], ncomp(h) - base, len(touched[n[1]]), df[n[1]]))
    rowsout.sort(key=lambda r: (-r[1], -r[2]))
    proj = nx.Graph()
    for x, cs in shared.items():
        if x in common: continue
        cl = sorted(cs)
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)): proj.add_edge(cl[i], cl[j])
    print(f"\n{name}: {len(ctr)} lineages; artefacts shared by 2+ lineages: {len(shared)} ({len(common)} ubiquitous, df>500, dropped); lineage components {base}")
    print("  lineage links:", sorted(proj.edges()))
    print("  cut first (artefact, components gained, lineages touching, families in corpus):")
    for r in rowsout[:8]: print("   ", r)
