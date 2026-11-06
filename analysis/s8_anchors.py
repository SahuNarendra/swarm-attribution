from common import *
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# How many anchored files does a lineage need before attribution with refusal is usable?
# Setting: the 6 controllers with 30+ families are "known". For each anchor budget k, train on k files
# per known lineage, score the rest of their files (known test) and all files of the 2 hidden lineages
# (unknown). Threshold fixed where unknown files are wrongly attributed at 5 percent, found on each run.

d = load("prep.pkl"); rows, X, ws = d["rows"], d["X"], d["ws"]
y_all = {i: rows[i]["label"] for i, r in enumerate(rows) if r["label"]}
cnt = Counter(y_all.values())
big = sorted(c for c, n in cnt.items() if n >= 30)
small = sorted(c for c, n in cnt.items() if 8 <= n < 30)
print("known candidates:", {c: cnt[c] for c in big}, "| unknown pool:", {c: cnt[c] for c in small}, flush=True)


def fit(idx):
    y = np.array([y_all[i] for i in idx])
    sc = StandardScaler().fit(X[idx])
    rf = RandomForestClassifier(300, random_state=0, class_weight="balanced", n_jobs=2).fit(sc.transform(X[idx]), y)
    tv = TfidfVectorizer(token_pattern=r"\S+", min_df=1, sublinear_tf=True).fit([ws[i] for i in idx])
    lr = LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced").fit(tv.transform([ws[i] for i in idx]), y)
    def predict(jdx):
        P = (rf.predict_proba(sc.transform(X[jdx])) + lr.predict_proba(tv.transform([ws[j] for j in jdx]))) / 2
        return rf.classes_[P.argmax(1)], P.max(1)
    return predict


rng = np.random.default_rng(0)
K = [3, 5, 10, 20, 40]
res = {k: [] for k in K}
for rep in range(20):
    unk = list(rng.choice(small, 2, replace=False))
    U = [i for i in y_all if y_all[i] in unk]
    for k in K:
        tr, te = [], []
        for c in big:
            idx = [i for i in y_all if y_all[i] == c]
            rng.shuffle(idx)
            tr += idx[:k]; te += idx[k:]
        pr = fit(tr)
        lab, s = pr(te); ok = lab == np.array([y_all[i] for i in te])
        _, su = pr(U)
        # threshold: 5 percent of unknown files attributed
        t = np.quantile(su, 0.95)
        a = s >= t
        res[k].append((a.mean(), ok[a].mean() if a.any() else np.nan, ok.mean(), t))
print("\nANCHOR BUDGET (6 known lineages, 2 hidden; threshold set so 5 percent of hidden-lineage files are wrongly attributed; 20 reps)")
print(f"{'anchors per lineage':>20} {'known attributed':>17} {'precision':>10} {'top-1 no refusal':>17} {'threshold':>10}")
for k in K:
    v = np.array(res[k], float); m = np.nanmean(v, 0); sd = np.nanstd(v, 0)
    print(f"{k:>20} {m[0]:>11.3f}+-{sd[0]:.2f} {m[1]:>6.3f}+-{sd[1]:.2f} {m[2]:>11.3f}+-{sd[2]:.2f} {m[3]:>10.2f}")
save(res, "s8_out.pkl")
