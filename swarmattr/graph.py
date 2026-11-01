"""Coordination graph. Bipartite actor to artefact (key, file, board). containment_order ranks artefacts by how much removing each one fragments the actors.
"""

from __future__ import annotations

import re
from collections import defaultdict

from .loader import Payload

# what survives redaction in the Swarm Traces release (checked 1 Nov 2026):
# consistent pseudonym markers [REDACTED:<type>:<n>], .bin and command file names,
# dataset paths, paste services, url hosts, PEM headers. Raw keys, hashes, jwts do not.
_MARKER_RE = re.compile(r"\[REDACTED:(?:destination|runtime_identifier|source_identifier|source_path|query_value|jwt_claim_identifier):\d+\]")
_FILE_RE = re.compile(r"\b(?:[\w\-]+\.bin|[\w\-]*cmd[\w\-]*\.(?:bin|txt|sh|py)|datasets?/[\w\-./]+)\b")
_HOST_RE = re.compile(r"https?://([A-Za-z0-9.\-]+)")
_PASTE_RE = re.compile(r"\b(?:pastebin|dpaste|hastebin|rentry|ghostbin)[\w.\-]*\b", re.I)
_KEY_RE = re.compile(r"-----BEGIN [A-Z ]+-----|[A-Fa-f0-9]{64}")

MARKER_TYPES = ("destination", "runtime_identifier", "source_identifier", "source_path", "query_value", "jwt_claim_identifier")


def shared_artefacts(p: Payload, kinds: tuple = ("marker", "file", "host", "paste", "key")) -> set[str]:
    t = p.text or ""
    out = set()
    if "marker" in kinds:
        out.update(_MARKER_RE.findall(t))
    if "file" in kinds:
        out.update("file:" + m for m in _FILE_RE.findall(t))
    if "host" in kinds:
        out.update("host:" + m.lower().rstrip(".") for m in _HOST_RE.findall(t))
    if "paste" in kinds:
        out.update("paste:" + m.lower() for m in _PASTE_RE.findall(t))
    if "key" in kinds:
        out.update("key:" + m for m in _KEY_RE.findall(t))
    return out


def build_graph(payloads: list[Payload], actor_of: dict[str, int], kinds: tuple = ("marker", "file", "host", "paste", "key"), min_actors: int = 2):
    # bipartite: actor nodes (kind=actor) and artefact nodes (kind=artefact); artefacts touched by < min_actors are dropped
    import networkx as nx

    g = nx.Graph()
    for a in set(actor_of.values()):
        g.add_node(("actor", a), kind="actor")
    touched: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for p in payloads:
        a = actor_of.get(p.rec_id)
        if a is None:
            continue
        for art in shared_artefacts(p, kinds):
            touched[art][a] += 1
    for art, actors in touched.items():
        if len(actors) < min_actors:
            continue
        n = ("artefact", art)
        g.add_node(n, kind="artefact")
        for a, w in actors.items():
            g.add_edge(("actor", a), n, weight=w)
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
