from common import *
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

# Which files should an investigator verify first? s10b and s8b pick anchors at random. Here the k anchors
# are chosen from the unlabelled pool by a label-free rule and the rest is scored as before.
#   random      : as in s10b / s8b
#   coverage    : greedy k-medoids on the combined feature distance (each new anchor is the file that most
#                 reduces the summed distance from pool files to their nearest anchor)
#   diversity   : farthest-first traversal (k-center) on the same distance
#   least sure  : the k pool files with the lowest maximum class probability under the defender fitted
#                 without the pool (the files the classifier is least able to place)
#   most sure   : the k pool files with the highest maximum class probability (contrast)
# Setting A (s10b): pool = the 27 signed G236 families, never in training; defender = other lineages.
# Setting B (s8b): every pair of 8+ lineages hidden in turn; the k anchors per known lineage are chosen by
# the rule from that lineage's files; threshold set so 5 percent of hidden-lineage files are attributed.

d = load("prep.pkl"); rows, X, ws = d["rows"], d["X"], d["ws"]
T = load("s5_out.pkl")["T"]
y_all = {i: rows[i]["label"] for i, r in enumerate(rows) if r["label"]}
cnt = Counter(y_all.values())
lin = sorted(c for c, n in cnt.items() if n >= 8)
big = sorted(c for c, n in cnt.items() if n >= 30)
L = [i for i in y_all if y_all[i] in lin]
S = [i for i in L if rows[i]["signed"]]
TR = [i for i in L if i not in S]

# combined distance over the labelled files: scaled style features (euclidean, rescaled to [0,1] by its
# 95th percentile) averaged with cosine distance on the work-signature tf-idf
sc_all = StandardScaler().fit(X[L])
Dx = euclidean_distances(sc_all.transform(X[L])); Dx = np.minimum(Dx / np.percentile(Dx[Dx > 0], 95), 1.0)
tv_all = TfidfVectorizer(token_pattern=r"\S+", min_df=1, sublinear_tf=True).fit([ws[i] for i in L])
Dw = cosine_distances(tv_all.transform([ws[i] for i in L]))
D = (Dx + Dw) / 2
pos = {i: k for k, i in enumerate(L)}


def fit(idx, seed):
    y = np.array([y_all[i] for i in idx])
    sc = StandardScaler().fit(X[idx])
    rf = RandomForestClassifier(300, random_state=seed, class_weight="balanced", n_jobs=2).fit(sc.transform(X[idx]), y)
    tv = TfidfVectorizer(token_pattern=r"\S+", min_df=1, sublinear_tf=True).fit([ws[i] for i in idx])
    lr = LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced").fit(tv.transform([ws[i] for i in idx]), y)
    def predict(jdx):
        P = (rf.predict_proba(sc.transform(X[jdx])) + lr.predict_proba(tv.transform([ws[j] for j in jdx]))) / 2
        return rf.classes_[P.argmax(1)], P.max(1)
    return predict


def pick(pool, k, rule, rng, conf=None):
    pool = list(pool)
    if k >= len(pool): return pool
    if rule == "random": return list(rng.choice(pool, size=k, replace=False))
    if rule in ("least sure", "most sure"):
        order = np.argsort(conf)
        if rule == "most sure": order = order[::-1]
        return [pool[j] for j in order[:k]]
    sub = D[np.ix_([pos[i] for i in pool], [pos[i] for i in pool])]
    chosen = []
    if rule == "coverage":
        best = np.full(len(pool), np.inf)
        for _ in range(k):
            gain = [(np.minimum(best, sub[j]).sum(), j) for j in range(len(pool)) if j not in chosen]
            j = min(gain)[1]; chosen.append(j); best = np.minimum(best, sub[j])
    elif rule == "diversity":
        chosen = [int(sub.sum(1).argmin())]
        while len(chosen) < k:
            dmin = sub[:, chosen].min(1); dmin[chosen] = -1
            chosen.append(int(dmin.argmax()))
    return [pool[j] for j in chosen]


RULES = ("random", "coverage", "diversity", "least sure", "most sure")
rng = np.random.default_rng(0)

# ---- Setting A: signed chain ----
pr0 = fit(TR, 0)
_, conf_S = pr0(S)
REPS = 20
print(f"SETTING A: signed G236 chain ({len(S)} files) as the pool; k anchors chosen by rule, rest attributed; "
      f"random averaged over {REPS} draws, other rules deterministic (3 seeds)")
print(f"{'k':>3} {'rule':>11} {'top-1 G236':>11} {'attributed T5':>14} {'precision T5':>13} {'wrong+confident T5':>19}")
outA = {}
for k in (1, 2, 3, 5):
    for rule in RULES:
        r = {x: [] for x in ("top1", "a5", "p5", "w5")}
        for rep in range(REPS if rule == "random" else 3):
            A = pick(S, k, rule, rng, conf_S)
            te = [i for i in S if i not in A]
            lab, s = fit(TR + A, rep)(te)
            ok = lab == "G236"; a = s >= T[0.05]
            r["top1"].append(ok.mean()); r["a5"].append(a.mean())
            r["p5"].append(ok[a].mean() if a.any() else np.nan); r["w5"].append(((~ok) & a).mean())
        m = {x: float(np.nanmean(v)) if np.isfinite(v).any() else float("nan") for x, v in ((x, np.array(v, float)) for x, v in r.items())}
        outA[(k, rule)] = m
        f = lambda x: "  none  " if m[x] != m[x] else f"{m[x]:.3f}"
        print(f"{k:>3} {rule:>11} {f('top1'):>11} {f('a5'):>14} {f('p5'):>13} {f('w5'):>19}", flush=True)

# ---- Setting B: hidden pair rotated, anchors per known lineage ----
K = [3, 5, 10]
pairs = list(combinations(lin, 2))
outB = {(k, rule): [] for k in K for rule in RULES}
print(f"\nSETTING B: hidden pair rotated ({len(pairs)} pairs); k anchors per known lineage chosen by rule; "
      f"threshold at 5 percent of hidden-lineage files attributed")
for pair in pairs:
    U = [i for i in y_all if y_all[i] in pair]
    known = [c for c in big if c not in pair]
    # confidence for 'least sure' / 'most sure': leave-lineage-out score, fitted on the other known lineages
    conf = {}
    for c in known:
        others = [i for i in y_all if y_all[i] in known and y_all[i] != c]
        _, s = fit(others, 0)([i for i in y_all if y_all[i] == c])
        conf[c] = s
    for rule in RULES:
        for draw in range(2 if rule == "random" else 1):
            for k in K:
                tr, te = [], []
                for c in known:
                    idx = [i for i in y_all if y_all[i] == c]
                    A = pick(idx, k, rule, rng, conf[c])
                    tr += A; te += [i for i in idx if i not in A]
                p = fit(tr, draw)
                lab, s = p(te); ok = lab == np.array([y_all[i] for i in te])
                _, su = p(U); t = np.quantile(su, 0.95); a = s >= t
                outB[(k, rule)].append((a.mean(), ok[a].mean() if a.any() else np.nan, ok.mean()))
    print(f"  pair {pair} done", flush=True)
print(f"\n{'k':>3} {'rule':>11} {'known attributed':>26} {'precision':>26} {'top-1 no refusal':>26}")
for k in K:
    for rule in RULES:
        v = np.array(outB[(k, rule)], float)
        cell = lambda j: f"{np.nanmean(v[:, j]):.3f} [{np.nanpercentile(v[:, j], 2.5):.3f}, {np.nanpercentile(v[:, j], 97.5):.3f}]"
        print(f"{k:>3} {rule:>11} {cell(0):>26} {cell(1):>26} {cell(2):>26}")
save({"A": outA, "B": outB}, "s13_out.pkl")
