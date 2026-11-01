"""Coordination graph. Bipartite actor to artefact (key, file, board). containment_order ranks artefacts by how much removing each one fragments the actors.
"""

from __future__ import annotations

import re
from collections import defaultdict

from .loader import Payload

_REPO_RE = re.compile(r"(datasets?/[A-Za-z0-9_\-./]+|[A-Za-z0-9_\-]+\.bin|[A-Za-z0-9_\-]+cmd[A-Za-z0-9_\-]*)")
_KEY_RE = re.compile(r"(-----BEGIN[^-]+-----|[A-Fa-f0-9]{64})")
_BOARD_RE = re.compile(r"(zz[A-Z0-9_]+|/artifactory/[A-Za-z0-9_\-./]+|discussion[s]?/\d+)")


def shared_artefacts(p: Payload) -> set[str]:
    t = p.text or ""
    out = set()
    out.update(_REPO_RE.findall(t))
    out.update(_KEY_RE.findall(t))
    out.update(_BOARD_RE.findall(t))
    return out


def build_graph(payloads: list[Payload], actor_of: dict[str, int]):
    # bipartite: actor nodes (kind=actor) and artefact nodes (kind=artefact)
    import networkx as nx

    g = nx.Graph()
    for a in set(actor_of.values()):
        g.add_node(("actor", a), kind="actor")
    for p in payloads:
        a = actor_of.get(p.rec_id)
        if a is None:
            continue
        for art in shared_artefacts(p):
            n = ("artefact", art)
            if n not in g:
                g.add_node(n, kind="artefact")
            if g.has_edge(("actor", a), n):
                g[("actor", a)][n]["weight"] += 1
            else:
                g.add_edge(("actor", a), n, weight=1)
    return g


def actor_projection(g):
    import networkx as nx
    actors = [n for n, d in g.nodes(data=True) if d.get("kind") == "actor"]
    return nx.bipartite.weighted_projected_graph(g, actors)


def _n_components(g, actors) -> int:
    import networkx as nx
    comps = 0
    seen = set()
    for a in actors:
        if a in seen:
            continue
        comps += 1
        seen.update(nx.node_connected_component(g, a))
    return comps


def containment_order(g) -> list[tuple]:
    # for each artefact: components among actors after removing it, betweenness, degree
    import networkx as nx
    actors = [n for n, d in g.nodes(data=True) if d.get("kind") == "actor"]
    arts = [n for n, d in g.nodes(data=True) if d.get("kind") == "artefact"]
    if not arts:
        return []
    base = _n_components(g, actors)
    bc = nx.betweenness_centrality(g, weight="weight")
    rows = []
    for n in arts:
        h = g.copy()
        h.remove_node(n)
        gain = _n_components(h, actors) - base
        rows.append((n[1], gain, round(bc[n], 4), g.degree(n)))
    rows.sort(key=lambda r: (-r[1], -r[2], -r[3]))
    return rows
