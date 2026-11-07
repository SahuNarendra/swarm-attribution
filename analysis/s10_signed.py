from common import *
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from swarmattr.loader import load_records, build_families

# The signed G236 chain as ground truth for the whole pipeline.
# The 27 signed families carry the G236 name, so every earlier run had them in training.
# Here they are removed from training (with any near-copy of them), the defender is trained on
# name labels only, and the signed families are attributed with refusal at the open-set thresholds from s5.

d = load("prep.pkl"); rows, X, ws = d["rows"], d["X"], d["ws"]
T = load("s5_out.pkl")["T"]
L = [i for i, r in enumerate(rows) if r["label"]]
cnt = Counter(rows[i]["label"] for i in L); ctrls = sorted(c for c, n in cnt.items() if n >= 8)
L = [i for i in L if rows[i]["label"] in ctrls]
S = [i for i in L if rows[i]["signed"]]

# near-copies of signed text among the other labelled families (character 5-grams, names already stripped in features;
# raw text is used here only to find copies)
fams = {f.rec_id: f for f in build_families(load_records(DATA)) if f.n_chars >= 200}
txt = [strip_names(fams[rows[i]["rec_id"]].text) for i in L]
cv = TfidfVectorizer(analyzer="char", ngram_range=(5, 5), min_df=2, sublinear_tf=True).fit(txt)
M = cv.transform(txt)
spos = [L.index(i) for i in S]
sim = cosine_similarity(M[spos], M).max(0)
near = [L[k] for k in range(len(L)) if L[k] not in S and sim[k] >= 0.8]
print("near-copies of signed text among other labelled families (cosine >= 0.8):", len(near), Counter(rows[i]["label"] for i in near))
TR = [i for i in L if i not in S and i not in near]
ytr = np.array([rows[i]["label"] for i in TR])
print("training families", len(TR), Counter(ytr).most_common())
print("G236 training families by type", Counter(rows[i]["ptype"] for i in TR if rows[i]["label"] == "G236"))
print("signed families", len(S), "types", Counter(rows[i]["ptype"] for i in S))


def fit(idx, y, seed):
    sc = StandardScaler().fit(X[idx])
    rf = RandomForestClassifier(300, random_state=seed, class_weight="balanced", n_jobs=2).fit(sc.transform(X[idx]), y)
    tv = TfidfVectorizer(token_pattern=r"\S+", min_df=2, sublinear_tf=True).fit([ws[i] for i in idx])
    lr = LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced").fit(tv.transform([ws[i] for i in idx]), y)
    def predict(jdx):
        P = (rf.predict_proba(sc.transform(X[jdx])) + lr.predict_proba(tv.transform([ws[j] for j in jdx]))) / 2
        return rf.classes_[P.argmax(1)], P.max(1)
    return predict


SEEDS = range(10)
res = {k: [] for k in ("top1", "att59", "prec59", "att66", "prec66")}
per_file = np.zeros(len(S))
where, smax, wrong_conf = Counter(), [], {"59": [], "66": []}
for seed in SEEDS:
    pr = fit(TR, ytr, seed)
    lab, s = pr(S)
    where.update(lab); smax.append(s.max())
    for name, t in (("59", T[0.10]), ("66", T[0.05])): wrong_conf[name].append(((lab != "G236") & (s >= t)).mean())
    ok = lab == "G236"; per_file += ok
    res["top1"].append(ok.mean())
    for name, t in (("59", T[0.10]), ("66", T[0.05])):
        a = s >= t
        res["att" + name].append(a.mean()); res["prec" + name].append(ok[a].mean() if a.any() else np.nan)

# control: are other controllers' Python files pulled to G236? (4-fold CV on the training set, same thresholds)
rng = np.random.default_rng(0)
ctrl = {k: [] for k in ("py_other_to_g236", "py_other_to_g236_conf66", "g236_cv_top1")}
TRa = np.array(TR)
for seed in range(5):
    for a_, b_ in StratifiedKFold(4, shuffle=True, random_state=seed).split(TRa, ytr):
        pr = fit(list(TRa[a_]), ytr[a_], seed)
        te = TRa[b_]; lab, s = pr(list(te))
        py_other = np.array([rows[i]["ptype"] == "python" and rows[i]["label"] != "G236" for i in te])
        g = np.array([rows[i]["label"] == "G236" for i in te])
        if py_other.any():
            ctrl["py_other_to_g236"].append((lab[py_other] == "G236").mean())
            ctrl["py_other_to_g236_conf66"].append(((lab == "G236") & (s >= T[0.05]))[py_other].mean())
        if g.any(): ctrl["g236_cv_top1"].append((lab[g] == "G236").mean())


def ci(v):
    v = np.array([x for x in v if x == x])
    if not len(v): return "none attributed in any seed"
    return f"{v.mean():.3f} (seed range {v.min():.3f} to {v.max():.3f})"


def boot(p, n=10000):
    # 95% bootstrap interval over the 27 files for a per-file rate p (mean over seeds)
    b = np.random.default_rng(1).choice(p, size=(n, len(p))).mean(1)
    return np.percentile(b, [2.5, 97.5])


print(f"\nSIGNED CHAIN AS GROUND TRUTH (27 signed families never in training, {len(near)} near-copies also removed; 10 seeds)")
print("top-1 attributed to G236 without refusal:", ci(res["top1"]), "95% CI over files", np.round(boot(per_file / len(SEEDS)), 3))
print(f"at T={T[0.10]:.3f} (10% strangers wrongly claimed): attributed {ci(res['att59'])}, precision {ci(res['prec59'])}")
print(f"at T={T[0.05]:.3f} (5% strangers wrongly claimed): attributed {ci(res['att66'])}, precision {ci(res['prec66'])}")
print("where the signed files go (top-1, all seeds):", where.most_common())
print(f"highest top-class score on any signed file: {max(smax):.3f}")
print("signed files confidently attributed to a WRONG lineage at T(10%):", ci(wrong_conf["59"]), " at T(5%):", ci(wrong_conf["66"]))
print(f"chance: 1 of {len(ctrls)} lineages = {1/len(ctrls):.3f}; G236 share of training labels = {(ytr=='G236').mean():.3f}")
print("\nCONTROL (4-fold CV on training families, 5 seeds)")
print("other controllers' Python files attributed to G236 (no refusal):", ci(ctrl["py_other_to_g236"]))
print("other controllers' Python files confidently attributed to G236 at T(5%):", ci(ctrl["py_other_to_g236_conf66"]))
print("unsigned G236 files attributed to G236 in CV (no refusal):", ci(ctrl["g236_cv_top1"]))
save(dict(res=res, ctrl=ctrl, per_file=per_file, near=near), "s10_out.pkl")
