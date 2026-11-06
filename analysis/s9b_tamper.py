from common import *
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from swarmattr.loader import load_records, build_families, label_controllers, Payload
from swarmattr.features import feature_matrix, CLUSTER_FEATURES, work_signature
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# Countermeasure to impersonation: does the file carry two styles?
# Features: style of the whole file, plus the distance between the style vectors of its first and
# second half, and the cosine between the work signatures of the two halves.
# Defender trains on clean files and on imitations it builds itself from training-fold files
# (any A dressed as any B). Tested on imitations of test-fold files it has never seen.

recs = load_records(DATA)
fam, y = [], []
for f in build_families(recs):
    if f.n_chars < 200: continue
    labs = label_controllers(f.text, f.tags)
    if len(labs) == 1: fam.append(strip_names(f.text)); y.append(labs[0])
cnt = Counter(y); keep = [k for k, c in enumerate(y) if cnt[c] >= 8]
fam = [fam[k] for k in keep]; y = np.array([y[k] for k in keep]); ctrls = sorted(set(y))


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


def vocab_lines(tokens, rng):
    out = []
    for t in tokens:
        kind, _, name = t.partition(":")
        if kind == "cmd": out.append(f"{name} --help > /dev/null 2>&1")
        elif kind == "imp": out.append(f"import {name}")
        elif kind == "fn": out.append(f"{name}()")
    rng.shuffle(out)
    return "\n".join(out)


def imitations(idx, pool, top, rng, moves):
    out = {m: [] for m in moves}
    for i in idx:
        A = y[i]; B = rng.choice([c for c in ctrls if c != A])
        for m in moves:
            x = fam[i]
            if m in ("splice", "both"): x = pool[B][rng.integers(len(pool[B]))] + "\n" + x
            if m in ("vocab", "both"): x = vocab_lines(top[B], rng) + "\n" + x
            out[m].append(x)
    return out


MOVES = ("splice", "vocab", "both")
rng = np.random.default_rng(0)
res = {m: [] for m in MOVES}
for seed in range(3):
    for tr, te in StratifiedKFold(4, shuffle=True, random_state=seed).split(fam, y):
        tv = TfidfVectorizer(token_pattern=r"\S+", min_df=2, sublinear_tf=True).fit([work_signature(Payload(rec_id="x", text=fam[i])) for i in tr])
        # B's characteristic tokens from a quick per-lineage tf-idf mean (attacker view, training files only)
        Wtr = tv.transform([work_signature(Payload(rec_id="x", text=fam[i])) for i in tr]).toarray()
        vocab = np.array(tv.get_feature_names_out())
        top = {}
        for c in ctrls:
            mc = Wtr[y[tr] == c].mean(0) - Wtr.mean(0)
            top[c] = list(vocab[np.argsort(-mc)[:12]])
        pool = {c: [fam[i] for i in tr if y[i] == c] for c in ctrls}
        im_tr = imitations(tr, pool, top, rng, MOVES)
        clean_tr = [fam[i] for i in tr]
        Xtr = tamper_features(clean_tr + sum((im_tr[m] for m in MOVES), []), tv)
        ytr = np.r_[np.zeros(len(clean_tr)), np.ones(len(clean_tr) * len(MOVES))]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed).fit(Xtr, ytr)
        clean_te = [fam[i] for i in te]
        s_clean = clf.predict_proba(tamper_features(clean_te, tv))[:, 1]
        thr = np.quantile(s_clean, 0.95)  # 5 percent false alarms on clean test files
        im_te = imitations(te, pool, top, rng, MOVES)
        for m in MOVES:
            s_im = clf.predict_proba(tamper_features(im_te[m], tv))[:, 1]
            auc = roc_auc_score(np.r_[np.zeros(len(s_clean)), np.ones(len(s_im))], np.r_[s_clean, s_im])
            res[m].append((auc, (s_im > thr).mean()))
print("TAMPER DETECTOR (two-style check; trained on self-made imitations of training files; 4-fold x 3 seeds)")
print(f"{'move':<8}{'AUC':>12}{'caught at 5% false alarms':>28}")
for m in MOVES:
    v = np.array(res[m]); print(f"{m:<8}{v[:,0].mean():>8.3f}+-{v[:,0].std():.2f}{v[:,1].mean():>20.3f}+-{v[:,1].std():.2f}")
save(res, "s9b_out.pkl")
