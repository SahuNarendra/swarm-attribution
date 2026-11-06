from common import *
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
d = load("prep.pkl"); rows, X, ws = d["rows"], d["X"], d["ws"]
idx = [i for i, r in enumerate(rows) if r["label"]]
y = np.array([rows[i]["label"] for i in idx])
res = {"style": [], "work": [], "avg": [], "both_right": [], "either_right": []}
for s in range(5):
    for tr, te in StratifiedKFold(4, shuffle=True, random_state=s).split(idx, y):
        itr = [idx[j] for j in tr]; ite = [idx[j] for j in te]
        sc = StandardScaler().fit(X[itr])
        rf = RandomForestClassifier(300, random_state=s, class_weight="balanced", n_jobs=2).fit(sc.transform(X[itr]), y[tr])
        tv = TfidfVectorizer(token_pattern=r"\S+", min_df=2, sublinear_tf=True).fit([ws[j] for j in itr])
        lr = LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced").fit(tv.transform([ws[j] for j in itr]), y[tr])
        P1 = rf.predict_proba(sc.transform(X[ite])); P2 = lr.predict_proba(tv.transform([ws[j] for j in ite]))
        a1 = rf.classes_[P1.argmax(1)] == y[te]; a2 = lr.classes_[P2.argmax(1)] == y[te]
        res["style"].append(a1.mean()); res["work"].append(a2.mean())
        res["avg"].append((rf.classes_[(P1 + P2).argmax(1)] == y[te]).mean())
        res["both_right"].append((a1 & a2).mean()); res["either_right"].append((a1 | a2).mean())
for k, v in res.items(): print(f"{k:<13} {np.mean(v):.3f} +- {np.std(v):.3f}")
