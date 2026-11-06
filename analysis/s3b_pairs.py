from common import *
import warnings
warnings.filterwarnings("ignore")
from itertools import combinations
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
d = load("prep.pkl"); rows, X, ws = d["rows"], d["X"], d["ws"]
a = load("arts.pkl"); arts, df = a["arts"], a["df"]
Z = StandardScaler().fit_transform(X)
W = normalize(TfidfVectorizer(token_pattern=r"\S+", min_df=3, sublinear_tf=True).fit_transform(ws))
Zn = normalize(Z)
types = np.array([r["ptype"] for r in rows]); lens = np.array([r["n_chars"] for r in rows])

def pairfeat(i, j, which=("style", "work", "art", "meta")):
    f = []
    if "style" in which:
        f += list(np.abs(Z[i] - Z[j])); f += [float(np.linalg.norm(Z[i] - Z[j])), float(Zn[i] @ Zn[j])]
    if "work" in which:
        f += [float(W[i].multiply(W[j]).sum())]
    if "art" in which:
        sh = [x for x in arts[i] & arts[j] if df[x] <= 50]
        f += [len(sh), sum(1 / math.log(df[x] + 1) for x in sh), float(bool(arts[i] & arts[j]))]
    if "meta" in which:
        f += [float(types[i] == types[j]), abs(math.log(lens[i] / lens[j]))]
    return f

L = [i for i, r in enumerate(rows) if r["label"]]
lab = {i: rows[i]["label"] for i in L}
ctrls = sorted(set(lab.values()))
pairs = list(combinations(L, 2))
yp = np.array([lab[i] == lab[j] for i, j in pairs])
print("labelled pairs:", len(pairs), "positive:", yp.sum(), flush=True)
SETS = {"style": ("style", "meta"), "work": ("work", "meta"), "artefacts": ("art", "meta"), "all": ("style", "work", "art", "meta")}
FE = {k: np.array([pairfeat(i, j, v) for i, j in pairs]) for k, v in SETS.items()}

def thr_at(scores, y, fpr):
    neg = np.sort(scores[~y]); return neg[int(math.ceil((1 - fpr) * len(neg))) - 1]

rng = np.random.default_rng(0)
res = {k: [] for k in SETS}
for rep in range(10):
    test_c = set(rng.choice(ctrls, size=len(ctrls) // 2, replace=False))
    tr = np.array([lab[i] not in test_c and lab[j] not in test_c for i, j in pairs])
    te = np.array([lab[i] in test_c and lab[j] in test_c for i, j in pairs])
    if yp[te].sum() == 0 or (~yp[te]).sum() == 0: continue
    for k in SETS:
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=rep).fit(FE[k][tr], yp[tr])
        s = m.predict_proba(FE[k][te])[:, 1]; yt = yp[te]
        row = [roc_auc_score(yt, s)]
        for fpr in (0.01, 0.05):
            t = thr_at(s, yt, fpr); pred = s > t
            tp = (pred & yt).sum(); fp = (pred & ~yt).sum()
            row += [tp / yt.sum(), tp / max(1, tp + fp)]
        res[k].append(row)
print("HELD-OUT CONTROLLERS (10 random halves; model never saw the test lineages)")
print(f"{'features':<10} {'AUC':>12} {'recall@1%FPR':>14} {'prec@1%FPR':>12} {'recall@5%FPR':>14} {'prec@5%FPR':>12}")
for k, v in res.items():
    v = np.array(v); m, s = v.mean(0), v.std(0)
    print(f"{k:<10} {m[0]:.3f}+-{s[0]:.3f} {m[1]:>8.3f}+-{s[1]:.3f} {m[2]:>6.3f}+-{s[2]:.3f} {m[3]:>8.3f}+-{s[3]:.3f} {m[4]:>6.3f}+-{s[4]:.3f}")

# signed anchor check: train with G236 held out entirely
sig = set(i for i in L if rows[i]["signed"])
tr = np.array([lab[i] != "G236" and lab[j] != "G236" for i, j in pairs])
m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(FE["all"][tr], yp[tr])
neg_tr = m.predict_proba(FE["all"][tr & ~yp])[:, 1]
for fpr in (0.01, 0.05):
    t = np.sort(neg_tr)[int(math.ceil((1 - fpr) * len(neg_tr))) - 1]
    ss = [k for k, (i, j) in enumerate(pairs) if i in sig and j in sig]
    so = [k for k, (i, j) in enumerate(pairs) if (i in sig) != (j in sig) and lab[i] != lab[j]]
    gs = [k for k, (i, j) in enumerate(pairs) if (i in sig) != (j in sig) and lab[i] == lab[j] == "G236"]
    p = lambda ks: (m.predict_proba(FE["all"][ks])[:, 1] > t).mean()
    print(f"SIGNED CHAIN (G236 never seen in training), threshold at {fpr:.0%} training FPR: signed-signed linked {p(ss):.3f} ({len(ss)} pairs); signed-to-other-G236 linked {p(gs):.3f} ({len(gs)}); signed-to-other-controller false links {p(so):.3f} ({len(so)})")
save(dict(SETS=SETS), "s3_meta.pkl")
