from common import *
import warnings
warnings.filterwarnings("ignore")
from itertools import combinations
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score
from sklearn.ensemble import HistGradientBoostingClassifier
from pairfeat import *
L = [i for i, r in enumerate(rows) if r["label"]]
lab = {i: rows[i]["label"] for i in L}; ctrls = sorted(set(lab.values()))
LP = np.array(list(combinations(L, 2))); yL = np.array([lab[i] == lab[j] for i, j in LP]); FL = feats(LP)
rng = np.random.default_rng(1)
R = {}
def add(name, k, pred, truth):
    R.setdefault(name, []).append((k, len(set(pred)), adjusted_rand_score(truth, pred)))
for rep in range(10):
    test_c = sorted(rng.choice(ctrls, size=len(ctrls) // 2, replace=False))
    trn_c = [c for c in ctrls if c not in test_c]
    # out-of-fold scores on training lineages to set thresholds honestly (inner split of training controllers)
    inner = []
    for f in range(3):
        ic = set(np.random.default_rng(rep * 10 + f).choice(trn_c, size=len(trn_c) // 2, replace=False))
        a_ = np.array([lab[i] in trn_c and lab[j] in trn_c and lab[i] not in ic and lab[j] not in ic for i, j in LP])
        b_ = np.array([lab[i] in ic and lab[j] in ic for i, j in LP])
        if (yL[b_] == 0).sum() == 0: continue
        mi = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=f).fit(FL[a_], yL[a_])
        inner += list(mi.predict_proba(FL[b_ & ~yL])[:, 1])
    inner = np.sort(inner)
    th = {f: inner[int(math.ceil((1 - f) * len(inner))) - 1] for f in (0.001, 0.01, 0.05)}
    tr = np.array([lab[i] in trn_c and lab[j] in trn_c for i, j in LP])
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=rep).fit(FL[tr], yL[tr])
    nodes = [i for i in L if lab[i] in test_c]; pos = {v: k for k, v in enumerate(nodes)}
    TP = np.array([(i, j) for i, j in LP if i in pos and j in pos]); s = m.predict_proba(feats(TP))[:, 1]
    E = np.array([(pos[i], pos[j]) for i, j in TP]); n = len(nodes); truth = [lab[i] for i in nodes]
    S = np.zeros((n, n)); S[E[:, 0], E[:, 1]] = s; S[E[:, 1], E[:, 0]] = s
    for f, t in th.items():
        k = s > t
        g = coo_matrix((np.ones(k.sum()), (E[k, 0], E[k, 1])), shape=(n, n))
        add(f"chain@{f}", len(test_c), connected_components(g, directed=False)[1], truth)
        G = nx.Graph(); G.add_nodes_from(range(n)); G.add_weighted_edges_from([(a, b, w) for (a, b), w in zip(E[k], s[k])])
        comm = nx.community.louvain_communities(G, weight="weight", seed=0)
        pred = np.zeros(n, int)
        for ci, c in enumerate(comm): pred[list(c)] = ci
        add(f"louvain@{f}", len(test_c), pred, truth)
        D = 1 - S; np.fill_diagonal(D, 0)
        add(f"avglink@{f}", len(test_c), fcluster(linkage(squareform(D, checks=False), "average"), 1 - t, "distance"), truth)
print("held-out: 5 true lineages per rep, 10 reps")
print(f"{'method':<16} {'groups found (median, range)':<30} {'ARI':>12}")
for k, v in R.items():
    v = np.array(v, float)
    print(f"{k:<16} {np.median(v[:,1]):>6.0f}  ({v[:,1].min():.0f}-{v[:,1].max():.0f}){'':<16} {v[:,2].mean():.3f}+-{v[:,2].std():.3f}")
save(R, "s4b_out.pkl")
