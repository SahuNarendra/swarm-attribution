from common import *
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from itertools import permutations
from swarmattr.loader import load_records, build_families, label_controllers, Payload
from swarmattr.features import feature_matrix, CLUSTER_FEATURES, work_signature
from swarmattr.adversary import perturb
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer

# Adaptive impersonation. The attacker in s9 makes one fixed move. Here the attacker has a SURROGATE
# defender (trained on B's and everyone's public files, i.e. the training fold, which the attacker can see
# since the artefacts are public), tries N candidate disguises of each file, and keeps the one the surrogate
# scores highest for B. The real defender is a different model (different seed, same training data) with the
# refusal threshold T and the s9b tamper check. This is the worst case for framing with black-box access.
# Candidate moves: splice with a random B file; vocab lines from B; both; each optionally after
# renaming A's own identifiers (adversary.perturb at 0.1) so A's work signature is weakened first.

T = 0.66
N_CAND = 16
recs = load_records(DATA)
fam, y = [], []
for f in build_families(recs):
    if f.n_chars < 200: continue
    labs = label_controllers(f.text, f.tags)
    if len(labs) == 1: fam.append(strip_names(f.text)); y.append(labs[0])
cnt = Counter(y); keep = [k for k, c in enumerate(y) if cnt[c] >= 8]
fam = [fam[k] for k in keep]; y = np.array([y[k] for k in keep]); ctrls = sorted(set(y))
print("families", len(fam), "lineages", len(ctrls), "candidates per file", N_CAND, flush=True)


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
    vocab = np.array(tv.get_feature_names_out())
    top = {c: [t for t in vocab[np.argsort(-lr.coef_[k])[:12]]] for k, c in enumerate(lr.classes_)}
    classes = rf.classes_
    def proba(texts):
        Xt, Wt = featurize(texts)
        return (rf.predict_proba(sc.transform(Xt)) + lr.predict_proba(tv.transform(Wt))) / 2
    return proba, classes, top, tv


def vocab_lines(tokens, rng):
    out = []
    for t in tokens:
        kind, _, name = t.partition(":")
        if kind == "cmd": out.append(f"{name} --help > /dev/null 2>&1")
        elif kind == "imp": out.append(f"import {name}")
        elif kind == "fn": out.append(f"{name}()")
    rng.shuffle(out)
    return "\n".join(out)


def candidates(t, B, pool, top, rng):
    out = []
    for _ in range(N_CAND):
        x = t
        if rng.random() < 0.5: x = perturb(Payload(rec_id="c", text=x), 0.1, seed=int(rng.integers(1 << 30))).text
        move = rng.choice(["splice", "vocab", "both"])
        if move in ("splice", "both"): x = pool[B][rng.integers(len(pool[B]))] + "\n" + x
        if move in ("vocab", "both"): x = vocab_lines(top[B], rng) + "\n" + x
        out.append(x)
    return out


# tamper check from s9b
def halves(t):
    lines = t.split("\n"); m = max(1, len(lines) // 2)
    return "\n".join(lines[:m]), "\n".join(lines[m:])


def tamper_features(texts, tv):
    pl = [Payload(rec_id=str(i), text=t) for i, t in enumerate(texts)]
    X, _ = feature_matrix(pl, CLUSTER_FEATURES)
    A, Bh = zip(*(halves(t) for t in texts))
    XA, _ = feature_matrix([Payload(rec_id="a", text=t) for t in A], CLUSTER_FEATURES)
    XB, _ = feature_matrix([Payload(rec_id="b", text=t) for t in Bh], CLUSTER_FEATURES)
    gap = np.abs(XA - XB) / (X.std(0) + 1e-9)
    WA = normalize(tv.transform([work_signature(Payload(rec_id="a", text=t)) for t in A]))
    WB = normalize(tv.transform([work_signature(Payload(rec_id="b", text=t)) for t in Bh]))
    cos = np.asarray(WA.multiply(WB).sum(1))
    return np.hstack([X, gap, gap.sum(1, keepdims=True), cos])


def fit_tamper(tr, top, tv, seed, rng):
    pool = {c: [fam[i] for i in tr if y[i] == c] for c in ctrls}
    clean, im = [], []
    for i in tr:
        A = y[i]; B = rng.choice([c for c in ctrls if c != A])
        clean.append(fam[i])
        for m in ("splice", "vocab", "both"):
            x = fam[i]
            if m in ("splice", "both"): x = pool[B][rng.integers(len(pool[B]))] + "\n" + x
            if m in ("vocab", "both"): x = vocab_lines(top[B], rng) + "\n" + x
            im.append(x)
    X = tamper_features(clean + im, tv)
    yy = np.r_[np.zeros(len(clean)), np.ones(len(im))]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed).fit(X, yy)
    return lambda texts: clf.predict_proba(tamper_features(texts, tv))[:, 1]


rng = np.random.default_rng(0)
res = {"fixed_both": [], "adaptive": []}
clean_lost = []
for seed in range(3):
    for tr, te in StratifiedKFold(4, shuffle=True, random_state=seed).split(fam, y):
        trt = [fam[i] for i in tr]
        proba_def, classes, top, tv = fit(trt, y[tr], seed)          # real defender
        proba_sur, classes_s, top_s, _ = fit(trt, y[tr], seed + 100)  # attacker's surrogate (same public data, other seed)
        tamper = fit_tamper(tr, top, tv, seed, rng)
        thr = np.quantile(tamper([fam[i] for i in te]), 0.95)
        pool = {c: [fam[i] for i in tr if y[i] == c] for c in ctrls}
        for A, B in permutations(ctrls, 2):
            src = [fam[i] for i in te if y[i] == A]
            if not src: continue
            bidx = list(classes_s).index(B)
            fixed, adapt = [], []
            for t in src:
                fixed.append(vocab_lines(top_s[B], rng) + "\n" + pool[B][rng.integers(len(pool[B]))] + "\n" + t)
                cand = candidates(t, B, pool, top_s, rng)
                ps = proba_sur(cand)[:, bidx]
                adapt.append(cand[int(ps.argmax())])
            for name, texts in (("fixed_both", fixed), ("adaptive", adapt)):
                P = proba_def(texts); lab = classes[P.argmax(1)]; s = P.max(1)
                flag = tamper(texts) > thr
                fc = (lab == B) & (s >= T)
                res[name].append(dict(A=A, B=B, framed=(lab == B).mean(), framed_conf=fc.mean(),
                                      framed_conf_unflagged=(fc & ~flag).mean(), caught_conf=((lab == A) & (s >= T)).mean(), flagged=flag.mean()))
        lab_c = classes[proba_def([fam[i] for i in te]).argmax(1)]
        clean_lost.append(((tamper([fam[i] for i in te]) > thr)).mean())

brng = np.random.default_rng(1)
def boot(v):
    v = np.asarray(v, float); m = v[brng.integers(0, len(v), size=(10000, len(v)))].mean(1)
    return f"{v.mean():.3f} [{np.percentile(m, 2.5):.3f}, {np.percentile(m, 97.5):.3f}]"
print(f"\nADAPTIVE IMPERSONATION (surrogate defender, {N_CAND} candidates per file, best kept; real defender with refusal T={T} and the tamper check at 5% false alarms; 56 pairs x 4 folds x 3 seeds; bootstrap over pairs)")
print(f"{'attacker':<12}{'framed as B':>14}{'confidently framed':>20}{'framed, not flagged':>21}{'caught as A':>13}{'flagged':>10}")
for name in ("fixed_both", "adaptive"):
    v = res[name]; pairs = sorted({(r["A"], r["B"]) for r in v})
    pf = lambda k: [np.mean([r[k] for r in v if (r["A"], r["B"]) == p]) for p in pairs]
    print(f"{name:<12}{boot(pf('framed')):>14}{boot(pf('framed_conf')):>20}{boot(pf('framed_conf_unflagged')):>21}{boot(pf('caught_conf')):>13}{boot(pf('flagged')):>10}")
print(f"clean test files flagged by the tamper check: {np.mean(clean_lost):.3f}")
v = res["adaptive"]; byB = Counter(); nB = Counter()
for r in v: byB[r["B"]] += r["framed_conf_unflagged"]; nB[r["B"]] += 1
print("adaptive, framed and not flagged, by target B:", {c: round(byB[c] / nB[c], 3) for c in ctrls})
save(dict(res=res, clean_lost=clean_lost), "s12_out.pkl")
