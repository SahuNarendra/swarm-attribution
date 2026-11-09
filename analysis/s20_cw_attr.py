"""s20: collusion.wiki attribution, mirroring the Swarm Traces tests. Lenses: style = the same 20 per-line structural
features (random forest); vocabulary = word TF-IDF (logistic regression), the prose stand-in for the work signature.
Ensemble = mean of the two probabilities. All splits are grouped by wiki page, so no page is in train and test.
(A) top-1 over names; (B) same name within one hour vs more than a day apart vs different names (vocabulary cosine);
(C) open set: 20 percent of names hidden, threshold set on reps 0-9 (5 percent of hidden units attributed),
    evaluated at that fixed threshold on reps 10-19; (D) anchors: hidden names given k units as anchors."""
from common import *
import warnings, collections
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from datetime import datetime

d = load("s19_cw_prep.pkl")
U, X = d["units"], d["X"]
txt = [u["text"] for u in U]; y = np.array([u["label"] for u in U]); grp = np.array([u["page"] for u in U])
N = len(U); print("units", N, "names", len(set(y)), flush=True)


def fit(idx, seed):
    sc = StandardScaler().fit(X[idx])
    rf = RandomForestClassifier(300, random_state=seed, class_weight="balanced", n_jobs=4).fit(sc.transform(X[idx]), y[idx])
    tv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000).fit([txt[i] for i in idx])
    lr = LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced").fit(tv.transform([txt[i] for i in idx]), y[idx])
    def pred(jdx):
        P1 = rf.predict_proba(sc.transform(X[jdx])); P2 = lr.predict_proba(tv.transform([txt[j] for j in jdx]))
        return rf.classes_, P1, P2
    return pred


# (A) top-1
acc = collections.defaultdict(list)
for s in range(3):
    for tr, te in StratifiedGroupKFold(4, shuffle=True, random_state=s).split(np.zeros(N), y, grp):
        cl, P1, P2 = fit(tr, s)(te)
        acc["style"].append((cl[P1.argmax(1)] == y[te]).mean()); acc["vocab"].append((cl[P2.argmax(1)] == y[te]).mean())
        acc["both"].append((cl[(P1 + P2).argmax(1)] == y[te]).mean())
        acc["majority"].append((y[te] == collections.Counter(y[tr]).most_common(1)[0][0]).mean())
print("\n(A) TOP-1, grouped by page, 4 folds x 3 seeds")
for k, v in acc.items(): print(f"  {k:<9} {np.mean(v):.3f} +- {np.std(v):.3f}")

# (B) same name: within one hour vs more than a day apart vs different names
tv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000).fit(txt)
V = tv.transform(txt)
T = np.array([datetime.fromisoformat(u["time"].replace("Z", "+00:00")).timestamp() for u in U])
rng = np.random.default_rng(0)
near, far, diff = [], [], []
by = collections.defaultdict(list)
for i, l in enumerate(y): by[l].append(i)
for l, idx in by.items():
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            if grp[i] == grp[j]: continue
            dt = abs(T[i] - T[j])
            if dt <= 3600: near.append((i, j))
            elif dt >= 86400: far.append((i, j))
for _ in range(20000):
    i, j = rng.integers(N, size=2)
    if y[i] != y[j] and grp[i] != grp[j]: diff.append((i, j))
cos = lambda P: np.array([V[i].multiply(V[j]).sum() for i, j in P])
cn, cf, cd = cos(near), cos(far), cos(diff)
auc = lambda a, b: roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])
print(f"\n(B) VOCABULARY COSINE, different pages only: same name within 1 h {cn.mean():.3f} (n={len(cn)}), "
      f"same name 1+ day apart {cf.mean():.3f} (n={len(cf)}), different names {cd.mean():.3f} (n={len(cd)})")
print(f"  AUC vs different names: within 1 h {auc(cn, cd):.3f}; 1+ day apart {auc(cf, cd):.3f}")

# (C) open set and (D) anchors
names = np.array(sorted(set(y)))
scores = {}
def run(rep, k_list=(0,)):
    r = np.random.default_rng(100 + rep)
    hid = set(r.choice(names, size=max(2, len(names) // 5), replace=False))
    known_idx = np.array([i for i in range(N) if y[i] not in hid]); hid_idx = np.array([i for i in range(N) if y[i] in hid])
    # split known by page into train/test
    gk = np.array(sorted(set(grp[known_idx]))); r.shuffle(gk); te_pages = set(gk[: len(gk) // 4])
    tr = np.array([i for i in known_idx if grp[i] not in te_pages]); te = np.array([i for i in known_idx if grp[i] in te_pages])
    out = {}
    for k in k_list:
        anch = []
        if k:
            for h in hid:
                idx = [i for i in hid_idx if y[i] == h]; pg = sorted({grp[i] for i in idx}); r.shuffle(pg)
                chosen = []
                for p in pg:
                    chosen += [i for i in idx if grp[i] == p]
                    if len(chosen) >= k: break
                anch += chosen[:k]
        anch_pages = {grp[i] for i in anch}
        trk = np.r_[tr, np.array(anch, int)] if anch else tr
        pr = fit(trk, rep)
        cl, P1, P2 = pr(te); P = (P1 + P2) / 2; lk, sk = cl[P.argmax(1)], P.max(1)
        rest = np.array([i for i in hid_idx if grp[i] not in anch_pages and i not in anch]) if k else hid_idx
        cl2, Q1, Q2 = pr(rest); Q = (Q1 + Q2) / 2; lh, sh = cl2[Q.argmax(1)], Q.max(1)
        out[k] = dict(known_ok=lk == y[te], known_s=sk, hid_lab=lh, hid_true=y[rest], hid_s=sh)
    return out

reps = [run(i) for i in range(10)]
Tfix = float(np.median([np.quantile(o[0]["hid_s"], 0.95) for o in reps]))
print(f"\n(C) OPEN SET: {max(2, len(names) // 5)} of {len(names)} names hidden; threshold from reps 0-9: T = {Tfix:.3f}")
res = collections.defaultdict(list)
for i in range(10, 20):
    o = run(i, k_list=(0, 1, 3, 5, 10))
    for k, m in o.items():
        a = m["known_s"] >= Tfix
        res[k].append(dict(att=a.mean(), prec=m["known_ok"][a].mean() if a.any() else np.nan, top1=m["known_ok"].mean(),
                           strang=(m["hid_s"] >= Tfix).mean() if k == 0 else np.nan,
                           hid_top1=(m["hid_lab"] == m["hid_true"]).mean() if k else np.nan,
                           hid_att=((m["hid_lab"] == m["hid_true"]) & (m["hid_s"] >= Tfix)).mean() if k else np.nan,
                           hid_wrong=((m["hid_lab"] != m["hid_true"]) & (m["hid_s"] >= Tfix)).mean()))
    print(f"  rep {i} done", flush=True)
f = lambda v: f"{np.nanmean(v):.3f} [{np.nanpercentile(v, 2.5):.3f}, {np.nanpercentile(v, 97.5):.3f}]" if np.isfinite(np.nanmean(v)) else "  -"
r0 = res[0]
print(f"  known attributed {f([x['att'] for x in r0])}, precision {f([x['prec'] for x in r0])}, top-1 {f([x['top1'] for x in r0])}, "
      f"hidden-name units attributed {f([x['strang'] for x in r0])}")
print("\n(D) ANCHORS for the hidden names (k units each, rest of their units on other pages)")
for k in (1, 3, 5, 10):
    v = res[k]
    print(f"  k={k:<2} hidden top-1 {f([x['hid_top1'] for x in v])}  attributed correctly at T {f([x['hid_att'] for x in v])}  "
          f"wrong and confident {f([x['hid_wrong'] for x in v])}")
print(f"  k=0  wrong and confident on hidden names {f([x['hid_wrong'] for x in r0])}")
pickle.dump(dict(acc=dict(acc), B=(cn.mean(), cf.mean(), cd.mean(), auc(cn, cd), auc(cf, cd), len(cn), len(cf)), T=Tfix, res=dict(res)),
            open(out("s20_cw_attr_out.pkl"), "wb"))
