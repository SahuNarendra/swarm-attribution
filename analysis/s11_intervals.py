from common import *
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Intervals for the headline numbers that s5, s9, s9b and s9c print as means only.
# s5: the open-set test is re-run with the same seeds, keeping each repetition's numbers at the two s5 thresholds.
# s9, s9c: bootstrap over the 56 ordered lineage pairs (each pair averaged over its 12 runs first).
# s9b: range over the 12 fold runs. All intervals are 2.5 to 97.5 percentiles.

B = 10000
brng = np.random.default_rng(1)


def pct(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return np.nanmean(v), np.percentile(v, 2.5), np.percentile(v, 97.5)


def boot_mean(v):
    v = np.asarray(v, float)
    m = v[brng.integers(0, len(v), size=(B, len(v)))].mean(1)
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


fmt = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"

# s5 open set, per repetition
d = load("prep.pkl"); rows, X, ws = d["rows"], d["X"], d["ws"]
T = load("s5_out.pkl")["T"]
L = [i for i, r in enumerate(rows) if r["label"]]
y_all = {i: rows[i]["label"] for i in L}
cnt = Counter(y_all.values()); ctrls = sorted(c for c, n in cnt.items() if n >= 8)
L = [i for i in L if y_all[i] in ctrls]


def fit(idx):
    y = np.array([y_all[i] for i in idx])
    sc = StandardScaler().fit(X[idx]); rf = RandomForestClassifier(300, random_state=0, class_weight="balanced", n_jobs=2).fit(sc.transform(X[idx]), y)
    tv = TfidfVectorizer(token_pattern=r"\S+", min_df=2, sublinear_tf=True).fit([ws[i] for i in idx])
    lr = LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced").fit(tv.transform([ws[i] for i in idx]), y)
    def predict(jdx):
        P = (rf.predict_proba(sc.transform(X[jdx])) + lr.predict_proba(tv.transform([ws[j] for j in jdx]))) / 2
        return rf.classes_[P.argmax(1)], P.max(1)
    return predict


rng = np.random.default_rng(0)
per = {r: [] for r in T}
for rep in range(20):
    unk = set(rng.choice(ctrls, size=2, replace=False))
    K = [i for i in L if y_all[i] not in unk]; U = [i for i in L if y_all[i] in unk]
    yk = np.array([y_all[i] for i in K])
    ks, kok, us = [], [], []
    for tr, te in StratifiedKFold(4, shuffle=True, random_state=rep).split(K, yk):
        pr = fit([K[j] for j in tr])
        lab_, s = pr([K[j] for j in te]); ks += list(s); kok += list(lab_ == yk[te])
        _, su = pr(U); us = list(su)
    ks, kok, us = map(np.array, (ks, kok, us))
    for r, t in T.items():
        a = ks >= t
        per[r].append((a.mean(), kok[a].mean() if a.any() else np.nan, (us >= t).mean()))
print("OPEN SET WITH REFUSAL (s5 setting, 20 repetitions)")
for r, t in T.items():
    v = np.array(per[r], float)
    print(f"T={t:.3f}: known attributed {fmt(pct(v[:, 0]))}, precision {fmt(pct(v[:, 1]))}, strangers wrongly attributed {fmt(pct(v[:, 2]))}")

# s9 and s9c, bootstrap over pairs
s9 = load("s9_out.pkl"); s9c = load("s9c_out.pkl")
print("\nIMPERSONATION (s9), bootstrap over 56 ordered pairs")
for m in ("none", "splice", "vocab", "both"):
    v = s9[m]; pairs = sorted({(r["A"], r["B"]) for r in v})
    by = {k: {p: [] for p in pairs} for k in ("framed", "framed_conf", "caught_conf")}
    for r in v:
        for k in by: by[k][(r["A"], r["B"])].append(r[k])
    cells = {k: boot_mean([np.mean(by[k][p]) for p in pairs]) for k in by}
    print(f"{m:<7} framed as B, no refusal {fmt(cells['framed'])} | confidently framed {fmt(cells['framed_conf'])} | confidently caught as A {fmt(cells['caught_conf'])}")
print("\nREFUSAL + TAMPER CHECK (s9c), bootstrap over 56 ordered pairs (pair = run index mod 56)")
for m in ("none", "splice", "vocab", "both"):
    v = s9c["res"][m]; n = len(v)
    assert n % 56 == 0
    pf = lambda k: [np.mean([v[j][k] for j in range(p, n, 56)]) for p in range(56)]
    print(f"{m:<7} confidently framed {fmt(boot_mean(pf('framed_conf')))} | framed and not flagged {fmt(boot_mean(pf('framed_conf_unflagged')))}")
print(f"clean files lost to tamper flags: {fmt(pct(s9c['clean_lost']))} (range over 12 runs)")

# s9b, range over folds
s9b = load("s9b_out.pkl")
print("\nTAMPER CHECK (s9b), range over 12 fold runs")
for m in ("splice", "vocab", "both"):
    v = np.array(s9b[m], float)
    print(f"{m:<7} AUC {fmt(pct(v[:, 0]))} | caught at 5% false alarms {fmt(pct(v[:, 1]))}")
