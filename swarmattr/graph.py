"""Coordination graph and containment ordering.

Edges link actors that share a channel, key or mailbox. Containment ordering
ranks nodes by centrality to show which channel or credential to cut first to
fragment the population.
"""

from __future__ import annotations

import re
from collections import defaultdict

from .loader import Payload

# crude shared-artefact signals; refined against the real data at sprint time
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
    """Return a networkx graph linking actors that touch the same artefact.

    actor_of maps payload rec_id -> actor/cluster id. Nodes are actors; an edge
    weight counts shared artefacts between two actors.
    """
    import networkx as nx

    artefact_to_actors: dict[str, set] = defaultdict(set)
    for p in payloads:
        a = actor_of.get(p.rec_id)
        if a is None:
            continue
        for art in shared_artefacts(p):
            artefact_to_actors[art].add(a)

    g = nx.Graph()
    g.add_nodes_from(set(actor_of.values()))
    for art, actors in artefact_to_actors.items():
        actors = list(actors)
        for i in range(len(actors)):
            for j in range(i + 1, len(actors)):
                u, v = actors[i], actors[j]
                if g.has_edge(u, v):
                    g[u][v]["weight"] += 1
                    g[u][v]["artefacts"].add(art)
                else:
                    g.add_edge(u, v, weight=1, artefacts={art})
    return g


def containment_order(g) -> list[tuple]:
    """Rank actors/channels to cut first, by betweenness centrality.
    Returns [(actor, score), ...] high to low. Stub ordering; refined at sprint."""
    import networkx as nx
    if g.number_of_edges() == 0:
        return [(n, 0.0) for n in g.nodes()]
    bc = nx.betweenness_centrality(g, weight="weight")
    return sorted(bc.items(), key=lambda kv: kv[1], reverse=True)
