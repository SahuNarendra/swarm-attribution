from common import *
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

d = load("prep.pkl")
rows, X, ws = d["rows"], d["X"], d["ws"]
L = [i for i, r in enumerate(rows) if r["label"]]

def run(idx, lens=("style", "work", "both"), seeds=(0, 1, 2, 3, 4), min_n=2):
    y = np.array([rows[i]["label"] for i in idx])
    cnt = Counter(y); keep = [k for k, i in enumerate(idx) if cnt[y[k]] >= min_n]
    idx = [idx[k] for k in keep]; y = y[keep]
    k = min(5, min(Counter(y).values()))
    out = {l: [] for l in lens}
    for s in seeds:
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=s).split(idx, y):
            itr = [idx[j] for j in tr]; ite = [idx[j] for j in te]
            sc = StandardScaler().fit(X[itr]); Xtr, Xte = sc.transform(X[itr]), sc.transform(X[ite])
            tv = TfidfVectorizer(token_pattern=r"\S+", min_df=2, sublinear_tf=True).fit([ws[j] for j in itr])
            Wtr, Wte = tv.transform([ws[j] for j in itr]), tv.transform([ws[j] for j in ite])
            if "style" in out:
                p = RandomForestClassifier(300, random_state=s, class_weight="balanced", n_jobs=2).fit(Xtr, y[tr]).predict(Xte)
                out["style"].append((p == y[te]).mean())
            if "work" in out:
                p = LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced").fit(Wtr, y[tr]).predict(Wte)
                out["work"].append((p == y[te]).mean())
            if "both" in out:
                Btr = hstack([csr_matrix(Xtr), Wtr]).tocsr(); Bte = hstack([csr_matrix(Xte), Wte]).tocsr()
                p = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(Btr, y[tr]).predict(Bte)
                out["both"].append((p == y[te]).mean())
    base = max(Counter(y).values()) / len(y)
    return len(idx), len(set(y)), base, {l: (np.mean(v), np.std(v)) for l, v in out.items()}

def show(name, res):
    n, c, base, o = res
    print(f"{name:<34} n={n:<4} controllers={c:<2} majority={base:.3f} random={1/c:.3f} | " +
          " | ".join(f"{l} {m:.3f}+-{s:.3f}" for l, (m, s) in o.items()), flush=True)

show("all labelled (>=2 per ctrl)", run(L))
show("controllers with >=8 families", run(L, min_n=8))
for t in ("other", "python", "json", "html"):
    sub = [i for i in L if rows[i]["ptype"] == t]
    if len(sub) >= 30: show(f"within type: {t}", run(sub, min_n=8))
