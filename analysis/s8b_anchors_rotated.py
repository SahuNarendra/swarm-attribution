from common import *
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# s8 with the hidden pair rotated. In s8 the hidden pair was always PHASEONE and SC4 (the only lineages with
# 8 to 29 files). Here every pair of the 8 lineages with 8+ files is hidden in turn (28 pairs, 2 draws each);
# the known lineages are the ones with 30+ files that are not hidden. Threshold set per run so 5 percent of
# hidden-lineage files are wrongly attributed. Intervals are 2.5 to 97.5 percentiles across the 56 runs.

d = load("prep.pkl"); rows, X, ws = d["rows"], d["X"], d["ws"]
y_all = {i: rows[i]["label"] for i, r in enumerate(rows) if r["label"]}
cnt = Counter(y_all.values())
lin = sorted(c for c, n in cnt.items() if n >= 8)
big = sorted(c for c, n in cnt.items() if n >= 30)
print("lineages:", {c: cnt[c] for c in lin}, "| known pool (30+):", big, flush=True)


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


rng = np.random.default_rng(0)
K = [3, 5, 10, 20, 40]
res = {k: [] for k in K}
pairs = list(combinations(lin, 2))
for pair in pairs:
    U = [i for i in y_all if y_all[i] in pair]
    known = [c for c in big if c not in pair]
    for draw in range(2):
        for k in K:
            tr, te = [], []
            for c in known:
                idx = [i for i in y_all if y_all[i] == c]
                rng.shuffle(idx)
                tr += idx[:k]; te += idx[k:]
            pr = fit(tr, draw)
            lab, s = pr(te); ok = lab == np.array([y_all[i] for i in te])
            _, su = pr(U)
            t = np.quantile(su, 0.95)
            a = s >= t
            res[k].append((a.mean(), ok[a].mean() if a.any() else np.nan, ok.mean(), len(known)))
print(f"\nANCHOR BUDGET, HIDDEN PAIR ROTATED ({len(pairs)} pairs x 2 draws; known = 30+ lineages not hidden, {min(r[3] for r in res[3])} to {max(r[3] for r in res[3])} per run)")
print(f"{'anchors':>8} {'known attributed':>26} {'precision':>26} {'top-1 no refusal':>26}")
for k in K:
    v = np.array(res[k], float)
    cell = lambda j: f"{np.nanmean(v[:, j]):.3f} [{np.nanpercentile(v[:, j], 2.5):.3f}, {np.nanpercentile(v[:, j], 97.5):.3f}]"
    print(f"{k:>8} {cell(0):>26} {cell(1):>26} {cell(2):>26}")
save(res, "s8b_out.pkl")
