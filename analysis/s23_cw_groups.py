"""s23: collusion.wiki, from pairwise similarity to groups, with a topic control.
Ground-truth groups: communities (Louvain, seed 0) of the co-editing graph, where two names are linked by the number of
wiki pages both edited. Behavioural profiles use ONLY text a name added on pages no other kept name edited, so shared
pages cannot carry the signal. Topic = the page family assigned by the collusion.wiki authors (pages.jsonl), so a
name's topic profile is its distribution over page families.
(A) Topic control for s22: co-editors vs non-co-editors within bins of topic similarity.
(B) Blind grouping: agglomerative clustering of behavioural profiles, adjusted Rand and purity against the co-editing
    communities; same for topic profiles alone, and for a random partition.
(C) Anchors: k names per community labelled, every other name assigned to the nearest community centroid; accuracy for
    behavioural profiles, topic profiles and chance."""
from common import *
import json, collections, warnings
warnings.filterwarnings("ignore")
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score, adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering

CW = os.path.join(ROOT, "data", "collusion-wiki")
fam = {json.loads(l)["page_id"]: json.loads(l)["page_family"] for l in open(os.path.join(CW, "pages.jsonl"))}
U = load("s19_cw_prep.pkl")["units"]
txt = [u["text"] for u in U]; lab = np.array([u["label"] for u in U]); pg = np.array([u["page"] for u in U])
V = normalize(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000).fit_transform(txt))
names = sorted(set(lab)); idx = {n: np.where(lab == n)[0] for n in names}
pages = {n: set(pg[idx[n]]) for n in names}
fams = sorted({fam.get(p, "unknown") for p in pg}); fi = {f: i for i, f in enumerate(fams)}
def topic(n, excl=()):
    v = np.zeros(len(fams))
    for i in idx[n]:
        if pg[i] not in excl: v[fi[fam.get(pg[i], "unknown")]] += 1
    return v / (np.linalg.norm(v) or 1)
def prof(n, excl):
    k = [i for i in idx[n] if pg[i] not in excl]
    return None if len(k) < 3 else normalize(np.asarray(V[k].mean(0)))[0]

# (A) topic control
rows = []
for a in range(len(names)):
    for b in range(a + 1, len(names)):
        A, B = names[a], names[b]; sh = pages[A] & pages[B]
        pa, pb = prof(A, sh), prof(B, sh)
        if pa is None or pb is None: continue
        rows.append((bool(sh), float(pa @ pb), float(topic(A, sh) @ topic(B, sh))))
R = np.array(rows, float)
print(f"(A) TOPIC CONTROL, shared pages excluded ({len(R)} pairs)")
for lo, hi in ((0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)):
    m = (R[:, 2] >= lo) & (R[:, 2] < hi); y = R[m, 0]
    if y.sum() >= 20 and (1 - y).sum() >= 20:
        print(f"  topic similarity {lo:.2f} to {hi:.2f}: co-editors {int(y.sum())}, others {int((1-y).sum())}; vocabulary AUC {roc_auc_score(y, R[m, 1]):.3f}; topic-only AUC {roc_auc_score(y, R[m, 2]):.3f}")
    else:
        print(f"  topic similarity {lo:.2f} to {hi:.2f}: too few pairs ({int(y.sum())} / {int((1-y).sum())})")
print(f"  all pairs: vocabulary AUC {roc_auc_score(R[:, 0], R[:, 1]):.3f}; topic-only AUC {roc_auc_score(R[:, 0], R[:, 2]):.3f}")

# ground-truth groups
G = nx.Graph(); G.add_nodes_from(names)
for a in range(len(names)):
    for b in range(a + 1, len(names)):
        w = len(pages[names[a]] & pages[names[b]])
        if w: G.add_edge(names[a], names[b], weight=w)
comm = nx.community.louvain_communities(G, weight="weight", seed=0)
gid = {n: i for i, c in enumerate(comm) for n in c}
# behavioural profiles from unshared pages only
own = {n: {p for p in pages[n] if sum(p in pages[m] for m in names) == 1} for n in names}
keep = [n for n in names if sum(pg[i] in own[n] for i in idx[n]) >= 3]
Bp = normalize(np.vstack([np.asarray(V[[i for i in idx[n] if pg[i] in own[n]]].mean(0)) for n in keep]))
Tp = normalize(np.vstack([topic(n, set(pages[n]) - own[n]) for n in keep]))
y = np.array([gid[n] for n in keep]); sizes = collections.Counter(y)
big = [g for g, c in sizes.items() if c >= 3]
mk = np.isin(y, big); keep2 = [n for n, m in zip(keep, mk) if m]; y2 = y[mk]; Bp2, Tp2 = Bp[mk], Tp[mk]
print(f"\nco-editing communities: {len(comm)} (sizes {sorted([len(c) for c in comm], reverse=True)[:10]} ...); names with 3+ units on unshared pages: {len(keep)}; in communities of 3+ such names: {len(keep2)} across {len(big)} communities")
rng = np.random.default_rng(0)
def pur(lab_, y_): return sum(collections.Counter(y_[lab_ == c]).most_common(1)[0][1] for c in set(lab_)) / len(y_)
print("(B) BLIND GROUPING against co-editing communities (profiles from unshared pages only)")
for k in (len(big), 2 * len(big)):
    lb = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(Bp2)
    lt = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(Tp2 + 1e-9)
    rr = [adjusted_rand_score(y2, rng.permutation(lb)) for _ in range(200)]
    print(f"  k={k}: behaviour ARI {adjusted_rand_score(y2, lb):.3f} purity {pur(lb, y2):.3f} | topic only ARI {adjusted_rand_score(y2, lt):.3f} purity {pur(lt, y2):.3f} | random ARI 95th pct {np.percentile(rr, 95):.3f}")
print("(C) ANCHOR NAMES PER COMMUNITY, others assigned to nearest community centroid (50 draws)")
for k in (1, 2):
    accB, accT, ch = [], [], []
    for d in range(50):
        r = np.random.default_rng(d); anc, rest = [], []
        for g in big:
            ii = list(np.where(y2 == g)[0]); r.shuffle(ii); anc += ii[:k]; rest += ii[k:]
        for P_, acc in ((Bp2, accB), (Tp2, accT)):
            cen = normalize(np.vstack([P_[[i for i in anc if y2[i] == g]].mean(0) for g in big]))
            acc.append((np.array(big)[(P_[rest] @ cen.T).argmax(1)] == y2[rest]).mean())
        ch.append(np.mean([sizes[g] for g in y2[rest]]) * 0 + 1 / len(big))
    print(f"  k={k}: behaviour {np.mean(accB):.3f} [{np.percentile(accB, 2.5):.3f}, {np.percentile(accB, 97.5):.3f}] | topic only {np.mean(accT):.3f} | chance {np.mean(ch):.3f}")
save(dict(R=R), "s23_out.pkl")
