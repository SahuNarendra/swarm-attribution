from common import *
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from itertools import permutations
from swarmattr.loader import load_records, build_families, label_controllers, Payload
from swarmattr.features import feature_matrix, CLUSTER_FEATURES, work_signature
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer

# Refusal and tamper check together, on the same imitations.
# Attribution defender and imitations as in s9 (all 56 ordered pairs, attacker vocab from the work-signature model).
# Tamper check as in s9b (trained on its own imitations of training files, threshold at 5 percent false alarms
# on clean test files). A file counts as confidently framed only if it is attributed to B above T and the
# tamper check does not flag it.

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


def fit_attr(texts, labels, seed):
    X, W = featurize(texts)
    sc = StandardScaler().fit(X)
    rf = RandomForestClassifier(300, random_state=seed, class_weight="balanced", n_jobs=2).fit(sc.transform(X), labels)
    tv = TfidfVectorizer(token_pattern=r"\S+", min_df=2, sublinear_tf=True).fit(W)
    lr = LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced").fit(tv.transform(W), labels)
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


def halves(t):
    lines = t.split("\n"); m = max(1, len(lines) // 2)
    return "\n".join(lines[:m]), "\n".join(lines[m:])


def tamper_features(texts, tv):
    pl = [Payload(rec_id=str(i), text=t) for i, t in enumerate(texts)]
    X, _ = feature_matrix(pl, CLUSTER_FEATURES)
    A, B = zip(*(halves(t) for t in texts))
    XA, _ = feature_matrix([Payload(rec_id="a", text=t) for t in A], CLUSTER_FEATURES)
    XB, _ = feature_matrix([Payload(rec_id="b", text=t) for t in B], CLUSTER_FEATURES)
    sd = X.std(0) + 1e-9
    gap = np.abs(XA - XB) / sd
    WA = normalize(tv.transform([work_signature(Payload(rec_id="a", text=t)) for t in A]))
    WB = normalize(tv.transform([work_signature(Payload(rec_id="b", text=t)) for t in B]))
    cos = np.asarray(WA.multiply(WB).sum(1))
    return np.hstack([X, gap, gap.sum(1, keepdims=True), cos])


def fit_tamper(tr, seed, rng):
    tv = TfidfVectorizer(token_pattern=r"\S+", min_df=2, sublinear_tf=True).fit([work_signature(Payload(rec_id="x", text=fam[i])) for i in tr])
    Wtr = tv.transform([work_signature(Payload(rec_id="x", text=fam[i])) for i in tr]).toarray()
    vocab = np.array(tv.get_feature_names_out())
    top = {c: list(vocab[np.argsort(-(Wtr[y[tr] == c].mean(0) - Wtr.mean(0)))[:12]]) for c in ctrls}
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


MOVES = ("none", "splice", "vocab", "both")
res = {m: [] for m in MOVES}
clean_lost = []
rng = np.random.default_rng(0)
for seed in range(3):
    for tr, te in StratifiedKFold(4, shuffle=True, random_state=seed).split(fam, y):
        pr, top = fit_attr([fam[i] for i in tr], y[tr], seed)
        tamper = fit_tamper(tr, seed, rng)
        clean_te = [fam[i] for i in te]
        s_clean = tamper(clean_te)
        thr = np.quantile(s_clean, 0.95)
        lab_c, p_c = pr(clean_te)
        ok = (lab_c == y[te]) & (p_c >= T)
        clean_lost.append(((s_clean > thr) & ok).sum() / max(ok.sum(), 1))
        pool = {c: [fam[i] for i in tr if y[i] == c] for c in ctrls}
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
                flag = tamper(texts) > thr
                fc = (lab == B) & (s >= T)
                res[m].append(dict(framed_conf=fc.mean(), framed_conf_unflagged=(fc & ~flag).mean(), flagged=flag.mean()))
print(f"\nREFUSAL + TAMPER CHECK (s9 imitations, all 56 ordered pairs, 4-fold x 3 seeds; T={T}; tamper threshold at 5% false alarms)")
print(f"{'move':<8}{'confidently framed':>20}{'framed and not flagged':>25}{'flagged':>10}")
for m in MOVES:
    v = res[m]; f = lambda k: np.mean([r[k] for r in v])
    print(f"{m:<8}{f('framed_conf'):>20.3f}{f('framed_conf_unflagged'):>25.3f}{f('flagged'):>10.3f}")
print(f"\nclean test files correctly and confidently attributed that the tamper check flags: {np.mean(clean_lost):.3f}")
save(dict(res=res, clean_lost=clean_lost), "s9c_out.pkl")
