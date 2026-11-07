from common import *
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from swarmattr.loader import load_records, build_families

# Follow-up to s10: k signed G236 families are given to the defender as anchors (added to training as G236),
# the remaining signed families are attributed with refusal. k = 0 is the s10 setting.

d = load("prep.pkl"); rows, X, ws = d["rows"], d["X"], d["ws"]
T = load("s5_out.pkl")["T"]
L = [i for i, r in enumerate(rows) if r["label"]]
cnt = Counter(rows[i]["label"] for i in L); ctrls = sorted(c for c, n in cnt.items() if n >= 8)
L = [i for i in L if rows[i]["label"] in ctrls]
S = [i for i in L if rows[i]["signed"]]
TR = [i for i in L if i not in S]

# how alike are the signed families to each other (character 5-grams of the raw text, names stripped)
fams = {f.rec_id: f for f in build_families(load_records(DATA)) if f.n_chars >= 200}
txt = [strip_names(fams[rows[i]["rec_id"]].text) for i in S + TR]
M = TfidfVectorizer(analyzer="char", ngram_range=(5, 5), min_df=2, sublinear_tf=True).fit_transform(txt)
C = cosine_similarity(M[:len(S)], M)
within = C[:, :len(S)][np.triu_indices(len(S), 1)]
to_g236 = C[:, len(S):][:, [k for k, i in enumerate(TR) if rows[i]["label"] == "G236"]].ravel()
to_other = C[:, len(S):][:, [k for k, i in enumerate(TR) if rows[i]["label"] != "G236"]].ravel()
print(f"text similarity, signed vs signed: median {np.median(within):.3f}, max {within.max():.3f}, share >= 0.8: {(within >= 0.8).mean():.3f}")
print(f"text similarity, signed vs unsigned G236: median {np.median(to_g236):.3f}; signed vs other lineages: median {np.median(to_other):.3f}")


def fit(idx, y, seed):
    sc = StandardScaler().fit(X[idx])
    rf = RandomForestClassifier(300, random_state=seed, class_weight="balanced", n_jobs=2).fit(sc.transform(X[idx]), y)
    tv = TfidfVectorizer(token_pattern=r"\S+", min_df=2, sublinear_tf=True).fit([ws[i] for i in idx])
    lr = LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced").fit(tv.transform([ws[i] for i in idx]), y)
    def predict(jdx):
        P = (rf.predict_proba(sc.transform(X[jdx])) + lr.predict_proba(tv.transform([ws[j] for j in jdx]))) / 2
        return rf.classes_[P.argmax(1)], P.max(1)
    return predict


import sys
DEDUP = "--dedup" in sys.argv
rng = np.random.default_rng(0)
REPS = 20
if DEDUP: print("\nDEDUP: test files with a near-copy (cosine >= 0.8) among the anchors are dropped")
print(f"\nSIGNED ANCHORS (k signed families added to training as G236; the other {len(S)}-k attributed; {REPS} random draws)")
print(f"{'k':>3} {'top-1 G236':>11} {'attributed T10':>15} {'precision T10':>14} {'attributed T5':>14} {'precision T5':>13} {'wrong+confident T5':>19}")
out = {}
for k in (0, 1, 2, 3, 5, 10):
    r = {x: [] for x in ("top1", "a10", "p10", "a5", "p5", "w5")}
    for rep in range(REPS if k else 3):
        A = list(rng.choice(S, size=k, replace=False)) if k else []
        te = [i for i in S if i not in A]
        if DEDUP and A:
            near = {i for i in te if max(C[S.index(i), S.index(a)] for a in A) >= 0.8}
            te = [i for i in te if i not in near]
        idx = TR + A; y = np.array([rows[i]["label"] for i in idx])
        lab, s = fit(idx, y, rep)(te)
        ok = lab == "G236"
        r["top1"].append(ok.mean())
        for nm, t in (("10", T[0.10]), ("5", T[0.05])):
            a = s >= t
            r["a" + nm].append(a.mean()); r["p" + nm].append(ok[a].mean() if a.any() else np.nan)
        r["w5"].append(((~ok) & (s >= T[0.05])).mean())
    m = {x: (np.nanmean(v) if np.isfinite(v).any() else np.nan, np.nanpercentile(v, 2.5) if np.isfinite(v).any() else np.nan, np.nanpercentile(v, 97.5) if np.isfinite(v).any() else np.nan) for x, v in ((x, np.array(v, float)) for x, v in r.items())}
    out[k] = m
    f = lambda x: "  none  " if m[x][0] != m[x][0] else f"{m[x][0]:.3f}"
    print(f"{k:>3} {f('top1'):>11} {f('a10'):>15} {f('p10'):>14} {f('a5'):>14} {f('p5'):>13} {f('w5'):>19}")
print("\n2.5 to 97.5 percentile across draws (top-1 / attributed T5 / precision T5):")
for k, m in out.items():
    print(f"k={k}: top-1 {m['top1'][1]:.3f} to {m['top1'][2]:.3f}; attributed {m['a5'][1]:.3f} to {m['a5'][2]:.3f}; precision {m['p5'][1]:.3f} to {m['p5'][2]:.3f}")
save(out, "s10b_dedup_out.pkl" if DEDUP else "s10b_out.pkl")
