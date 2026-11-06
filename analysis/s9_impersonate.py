from common import *
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from itertools import permutations
from swarmattr.loader import load_records, build_families, label_controllers, Payload
from swarmattr.features import feature_matrix, CLUSTER_FEATURES, work_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Impersonation. A file written by lineage A is dressed up to look like lineage B.
# Three attacker moves, all black box (the attacker only needs B's public artefacts):
#   splice  - prepend one of B's files to A's file (wear B's clothes)
#   vocab   - add lines that use B's most characteristic commands and helper names
#   both    - splice then vocab
# Measured per ordered pair (A, B): framed (attributed to B), caught (attributed to A), refused (below threshold).
# Defender trains on clean files only (test fold of A is never seen). Threshold T from the open-set test.

T = 0.66
recs = load_records(DATA)
fam, y = [], []
for f in build_families(recs):
    if f.n_chars < 200: continue
    labs = label_controllers(f.text, f.tags)
    if len(labs) == 1: fam.append(strip_names(f.text)); y.append(labs[0])
cnt = Counter(y); keep = [k for k, c in enumerate(y) if cnt[c] >= 8]
fam = [fam[k] for k in keep]; y = np.array([y[k] for k in keep]); ctrls = sorted(set(y))
print("families", len(fam), "lineages", len(ctrls), flush=True)


def featurize(texts):
    pl = [Payload(rec_id=str(i), text=t) for i, t in enumerate(texts)]
    X, _ = feature_matrix(pl, CLUSTER_FEATURES)
    return X, [work_signature(p) for p in pl]


def fit(texts, labels, seed):
    X, W = featurize(texts)
    sc = StandardScaler().fit(X)
    rf = RandomForestClassifier(300, random_state=seed, class_weight="balanced", n_jobs=2).fit(sc.transform(X), labels)
    tv = TfidfVectorizer(token_pattern=r"\S+", min_df=2, sublinear_tf=True).fit(W)
    lr = LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced").fit(tv.transform(W), labels)
    # B's most characteristic tokens, from the work-signature model, for the vocab move
    vocab = np.array(tv.get_feature_names_out())
    top = {c: [t for t in vocab[np.argsort(-lr.coef_[k])[:12]]] for k, c in enumerate(lr.classes_)}
    def predict(texts):
        Xt, Wt = featurize(texts)
        P = (rf.predict_proba(sc.transform(Xt)) + lr.predict_proba(tv.transform(Wt))) / 2
        return rf.classes_[P.argmax(1)], P.max(1)
    return predict, top


def vocab_lines(tokens, rng):
    out = []
    for t in tokens:
        kind, _, name = t.partition(":")
        if kind == "cmd": out.append(f"{name} --help > /dev/null 2>&1")
        elif kind == "imp": out.append(f"import {name}")
        elif kind == "fn": out.append(f"{name}()")
    rng.shuffle(out)
    return "\n".join(out)


MOVES = ("none", "splice", "vocab", "both")
res = {m: [] for m in MOVES}
rng = np.random.default_rng(0)
for seed in range(3):
    for tr, te in StratifiedKFold(4, shuffle=True, random_state=seed).split(fam, y):
        pr, top = fit([fam[i] for i in tr], y[tr], seed)
        pool = {c: [fam[i] for i in tr if y[i] == c] for c in ctrls}   # B's public files, as seen by the attacker
        for A, B in permutations(ctrls, 2):
            src = [fam[i] for i in te if y[i] == A]
            if not src: continue
            for m in MOVES:
                texts = []
                for t in src:
                    x = t
                    if m in ("splice", "both"): x = pool[B][rng.integers(len(pool[B]))] + "\n" + x
                    if m in ("vocab", "both"): x = vocab_lines(top[B], rng) + "\n" + x
                    texts.append(x)
                lab, s = pr(texts)
                conf = s >= T
                res[m].append(dict(A=A, B=B, n=len(src),
                                   framed=(lab == B).mean(), caught=(lab == A).mean(),
                                   framed_conf=((lab == B) & conf).mean(), caught_conf=((lab == A) & conf).mean(), refused=(~conf).mean()))
print(f"\nIMPERSONATION (8 lineages, all 56 ordered pairs, 4-fold x 3 seeds; threshold T={T} from the open-set test)")
print(f"{'move':<8}{'framed as B':>14}{'caught as A':>14}{'framed, confident':>19}{'caught, confident':>19}{'refused':>10}")
for m in MOVES:
    v = res[m]
    f = lambda k: np.mean([r[k] for r in v])
    print(f"{m:<8}{f('framed'):>14.3f}{f('caught'):>14.3f}{f('framed_conf'):>19.3f}{f('caught_conf'):>19.3f}{f('refused'):>10.3f}")
# which targets are easiest to impersonate, which sources are hardest to hide
v = res["both"]
byB = Counter(); byA = Counter(); nB = Counter(); nA = Counter()
for r in v:
    byB[r["B"]] += r["framed_conf"]; nB[r["B"]] += 1; byA[r["A"]] += r["caught_conf"]; nA[r["A"]] += 1
print("\nconfident framing rate by target B (move=both):", {c: round(byB[c] / nB[c], 3) for c in ctrls})
print("confident caught rate by source A (move=both):", {c: round(byA[c] / nA[c], 3) for c in ctrls})
save(res, "s9_out.pkl")
