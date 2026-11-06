from common import *
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
d = load("prep.pkl"); rows, X, ws = d["rows"], d["X"], d["ws"]
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

# OPEN-SET TEST: 2 controllers hidden as "unknown lineages"; the rest known. Can we attribute known ones and refuse unknown ones?
rng = np.random.default_rng(0)
known_s, known_ok, unk_s = [], [], []
for rep in range(20):
    unk = set(rng.choice(ctrls, size=2, replace=False))
    K = [i for i in L if y_all[i] not in unk]; U = [i for i in L if y_all[i] in unk]
    yk = np.array([y_all[i] for i in K])
    for tr, te in StratifiedKFold(4, shuffle=True, random_state=rep).split(K, yk):
        pr = fit([K[j] for j in tr])
        lab_, s = pr([K[j] for j in te]); known_s += list(s); known_ok += list(lab_ == yk[te])
        _, su = pr(U); unk_s += list(su / 4)  # weight so each unknown family counts once across 4 folds
        unk_s[-len(U):] = list(su)
known_s, known_ok, unk_s = map(np.array, (known_s, known_ok, unk_s))
print(f"open-set test: {len(known_s)} known-lineage predictions, {len(unk_s)} unknown-lineage predictions (20 reps)")
print(f"{'threshold':>9} {'known: attributed':>18} {'known: precision':>17} {'unknown: wrongly attributed':>28}")
rows_out = []
for t in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    a = known_s >= t
    rows_out.append((t, a.mean(), known_ok[a].mean() if a.any() else np.nan, (unk_s >= t).mean()))
    print(f"{t:>9.1f} {a.mean():>18.3f} {known_ok[a].mean():>17.3f} {(unk_s >= t).mean():>28.3f}")
def t_at(rate):
    for t in np.linspace(0, 1, 201):
        if (unk_s >= t).mean() <= rate: return t
T = {r: t_at(r) for r in (0.05, 0.10)}
for r, t in T.items():
    a = known_s >= t
    print(f"threshold where unknown lineages are wrongly attributed {r:.0%}: t={t:.3f} -> known attributed {a.mean():.3f}, precision {known_ok[a].mean():.3f}")

# CORPUS: attribute all 53,405 families to the known lineages, with refusal
pr = fit(L)
allidx = np.arange(len(rows)); lab_, s = pr(list(allidx))
types = np.array([r["ptype"] for r in rows])
for r, t in T.items():
    acc = (s >= t) & ~np.isin(allidx, L)
    print(f"\nCORPUS (unlabelled families only) @ unknown-false-attribution {r:.0%} (t={t:.3f}): attributed {acc.sum()} of {(~np.isin(allidx, L)).sum()} unlabelled families ({acc.sum()/(~np.isin(allidx, L)).sum():.2%}); expected false attributions if all others were unknown lineages ~{int(r * (~np.isin(allidx, L)).sum())}")
    print("  by lineage:", Counter(lab_[acc]).most_common())
    print("  by type   :", {t_: f"{acc[types==t_].mean():.1%}" for t_ in ("other","python","json","html","shell")})
save(dict(curve=rows_out, T=T, lab=lab_, s=s), "s5_out.pkl")
