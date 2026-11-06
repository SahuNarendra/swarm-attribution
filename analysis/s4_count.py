from common import *
from pairfeat import *
from itertools import combinations
from collections import Counter, defaultdict
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
# candidate pairs over the corpus
cand = set()
for t in np.unique(types):
    ix = np.where(types == t)[0]
    if len(ix) < 3: continue
    k = min(11, len(ix))
    for M, met in ((Z[ix], "euclidean"), (W[ix], "cosine")):
        nn = NearestNeighbors(n_neighbors=k, metric=met, algorithm="brute", n_jobs=2).fit(M)
        _, nb = nn.kneighbors(M)
        for r, row in enumerate(nb):
            for c in row[1:]:
                i, j = ix[r], ix[c]; cand.add((min(i, j), max(i, j)))
print(f"knn candidates: {len(cand)}  ({time.time()-t0:.0f}s)", flush=True)
inv = defaultdict(list)
for i, s in enumerate(arts):
    for x in s:
        if x in rare: inv[x].append(i)
for x, ids in inv.items():
    if len(ids) >= 2:
        for i, j in combinations(sorted(ids), 2): cand.add((i, j))
P = np.array(sorted(cand)); print(f"all candidates: {len(P)}  ({time.time()-t0:.0f}s)", flush=True)
F = feats(P); print(f"features done ({time.time()-t0:.0f}s)", flush=True)

# labelled pairs for training
L = [i for i, r in enumerate(rows) if r["label"]]
lab = {i: rows[i]["label"] for i in L}; ctrls = sorted(set(lab.values()))
LP = np.array(list(combinations(L, 2))); yL = np.array([lab[i] == lab[j] for i, j in LP]); FL = feats(LP)

def components(edges, n):
    if len(edges) == 0: return n, np.arange(n)
    g = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)

rng = np.random.default_rng(1)
val, counts = [], []
for rep in range(10):
    test_c = set(rng.choice(ctrls, size=len(ctrls) // 2, replace=False))
    tr = np.array([lab[i] not in test_c and lab[j] not in test_c for i, j in LP])
    te_nodes = [i for i in L if lab[i] in test_c]
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=rep).fit(FL[tr], yL[tr])
    # threshold: 1% false-link rate on training negatives from OTHER lineages
    negs = m.predict_proba(FL[tr & ~yL])[:, 1]; thr = np.sort(negs)[int(math.ceil(0.99 * len(negs))) - 1]
    # validation: recover held-out controllers from their families alone
    pos = {v: k for k, v in enumerate(te_nodes)}
    TP = np.array([(pos[i], pos[j]) for i, j in LP if i in pos and j in pos])
    TF = feats(np.array([(te_nodes[a], te_nodes[b]) for a, b in TP]))
    keep = m.predict_proba(TF)[:, 1] > thr
    nc, cl = components(TP[keep], len(te_nodes))
    truth = np.array([lab[i] for i in te_nodes])
    sizes = Counter(cl); multi = [c for c, n in sizes.items() if n >= 2]
    pur = np.mean([Counter(truth[cl == c]).most_common(1)[0][1] / sizes[c] for c in multi]) if multi else float("nan")
    cov = np.mean([sizes[c] >= 2 for c in cl])
    val.append((len(test_c), len(multi), nc, pur, cov))
    # corpus count
    s = m.predict_proba(F)[:, 1]
    nc2, cl2 = components(P[s > thr], N)
    sz = Counter(cl2); mult = sum(1 for n in sz.values() if n >= 2)
    covc = sum(n for n in sz.values() if n >= 2) / N
    big5 = sorted(sz.values(), reverse=True)[:5]
    counts.append((mult, covc, nc2, big5))
    print(f"rep {rep}: thr={thr:.3f} | held-out {len(test_c)} controllers -> {len(multi)} linked groups, purity {pur:.2f}, coverage {cov:.2f} | corpus: {mult} linked lineages covering {covc:.1%} of families, largest {big5}", flush=True)
v = np.array([x[:4] for x in val], float); c = np.array([x[:2] for x in counts], float)
print("\nVALIDATION (held-out controllers): true", v[:, 0].mean(), "| recovered groups median", np.median(v[:, 1]), "range", v[:, 1].min(), "-", v[:, 1].max(), "| purity", f"{np.nanmean(v[:,3]):.2f}")
print("CORPUS linked lineages: median", np.median(c[:, 0]), "5-95%", np.percentile(c[:, 0], [5, 95]), "| coverage median", f"{np.median(c[:,1]):.3f}")
save(dict(val=val, counts=counts), "s4_out.pkl")
print(f"done {time.time()-t0:.0f}s")
