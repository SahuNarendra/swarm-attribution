from common import *
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from swarmattr.loader import load_records, build_families, KNOWN_CONTROLLERS, _CANON

# Three checks raised in review.
# (a) Label noise: labels come from substring matches, so a name like BE90 can match inside a hex string.
#     Re-label (rule "hex": ignore names inside runs of 8+ hex characters; rule "boundary": also require no
#     letter or digit on either side, which drops handle variants too) and re-run the
#     two-lens ensemble (as s2b) and the signed-chain test (as s10, without the near-copy filter).
# (c) Document type alone: a classifier that sees only the 6-way document type, same folds as s2b.

d = load("prep.pkl"); rows, X, ws = d["rows"], d["X"], d["ws"]
fams = {f.rec_id: f for f in build_families(load_records(DATA)) if f.n_chars >= 200}
pats = [(re.compile(r"(?<![0-9a-z])" + re.escape(nm.lower()) + r"(?![0-9a-z])"), _CANON.get(nm.lower(), nm)) for nm in KNOWN_CONTROLLERS]


HEXRUN = re.compile(r"[0-9a-f]{8,}")


def strict_labels(text, tags, rule):
    hay = (text or "").lower() + " " + (tags or "").lower()
    if rule == "hex":  # drop a name only where it sits inside a run of 8+ hex characters (hashes, keys, ids)
        hay = HEXRUN.sub(" ", hay)
        found = [(nm in hay) for nm in (x.lower() for x in KNOWN_CONTROLLERS)]
        out = []
        for ok, nm in zip(found, KNOWN_CONTROLLERS):
            c = _CANON.get(nm.lower(), nm)
            if ok and c not in out: out.append(c)
        return out
    out = []
    for p, nm in pats:
        if p.search(hay) and nm not in out: out.append(nm)
    return out


RULE = sys.argv[1] if len(sys.argv) > 1 else "hex"   # "hex" (main) or "boundary" (stricter stress test)
print("label rule:", RULE)
clean = []
for r in rows:
    f = fams[r["rec_id"]]
    labs = strict_labels(f.text, f.tags, RULE)
    clean.append(labs[0] if len(labs) == 1 else None)
old = Counter(r["label"] for r in rows if r["label"]); new = Counter(c for c in clean if c)
changed = sum(1 for r, c in zip(rows, clean) if r["label"] != c)
print("labelled families: substring", sum(old.values()), "| boundary", sum(new.values()), "| changed", changed)
print("lost per lineage:", {k: old[k] - new.get(k, 0) for k in old if old[k] != new.get(k, 0)})
print("gained per lineage:", {k: new[k] - old.get(k, 0) for k in new if new[k] > old.get(k, 0)})


def cv(labels, use_type=False):
    idx = [i for i, l in enumerate(labels) if l]
    y = np.array([labels[i] for i in idx])
    acc, base = [], []
    T = np.array([[rows[i]["ptype"]] for i in range(len(rows))])
    for s in range(5):
        for tr, te in StratifiedKFold(4, shuffle=True, random_state=s).split(idx, y):
            itr = [idx[j] for j in tr]; ite = [idx[j] for j in te]
            if use_type:
                oh = OneHotEncoder(handle_unknown="ignore").fit(T[itr])
                m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(oh.transform(T[itr]), y[tr])
                acc.append((m.predict(oh.transform(T[ite])) == y[te]).mean())
            else:
                sc = StandardScaler().fit(X[itr])
                rf = RandomForestClassifier(300, random_state=s, class_weight="balanced", n_jobs=2).fit(sc.transform(X[itr]), y[tr])
                tv = TfidfVectorizer(token_pattern=r"\S+", min_df=2, sublinear_tf=True).fit([ws[j] for j in itr])
                lr = LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced").fit(tv.transform([ws[j] for j in itr]), y[tr])
                P = rf.predict_proba(sc.transform(X[ite])) + lr.predict_proba(tv.transform([ws[j] for j in ite]))
                acc.append((rf.classes_[P.argmax(1)] == y[te]).mean())
            base.append((y[te] == Counter(y[tr]).most_common(1)[0][0]).mean())
    return np.mean(acc), np.std(acc), np.mean(base), len(idx), len(set(y))


orig = [r["label"] for r in rows]
a, s, b, n, k = cv(orig, use_type=True)
print(f"\n(c) DOCUMENT TYPE ONLY, original labels: {a:.3f} +- {s:.3f} (majority {b:.3f}; {n} families, {k} lineages)")
a, s, b, n, k = cv(orig)
print(f"    two lenses, original labels:           {a:.3f} +- {s:.3f} (majority {b:.3f})")
a, s, b, n, k = cv(clean)
print(f"\n(a) two lenses, cleaned labels:          {a:.3f} +- {s:.3f} (majority {b:.3f}; {n} families, {k} lineages)")
a2, s2, b2, *_ = cv(clean, use_type=True)
print(f"    document type only, cleaned labels:   {a2:.3f} +- {s2:.3f}")

# signed chain with cleaned labels: train on cleaned labels (8+ files), never on signed families
T5 = load("s5_out.pkl")["T"][0.05]
cnt = Counter(c for c in clean if c); ctrls = {c for c, m in cnt.items() if m >= 8}
S = [i for i, r in enumerate(rows) if r["signed"]]
TR = [i for i, c in enumerate(clean) if c in ctrls and not rows[i]["signed"]]
ytr = np.array([clean[i] for i in TR])
top, att, wrongc, smax = [], [], [], []
for seed in range(10):
    sc = StandardScaler().fit(X[TR])
    rf = RandomForestClassifier(300, random_state=seed, class_weight="balanced", n_jobs=2).fit(sc.transform(X[TR]), ytr)
    tv = TfidfVectorizer(token_pattern=r"\S+", min_df=2, sublinear_tf=True).fit([ws[i] for i in TR])
    lr = LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced").fit(tv.transform([ws[i] for i in TR]), ytr)
    P = (rf.predict_proba(sc.transform(X[S])) + lr.predict_proba(tv.transform([ws[i] for i in S]))) / 2
    lab, sm = rf.classes_[P.argmax(1)], P.max(1)
    top.append((lab == "G236").mean()); att.append((sm >= T5).mean()); wrongc.append(((lab != "G236") & (sm >= T5)).mean()); smax.append(sm.max())
print(f"\n(a) signed chain, cleaned labels, {len(S)} signed families, 10 seeds: top-1 G236 {np.mean(top):.3f}, "
      f"attributed at T={T5:.2f} {np.mean(att):.3f}, wrongly and confidently {np.mean(wrongc):.3f}, max score {max(smax):.3f}")
save(dict(clean=clean), f"s15_{RULE}_out.pkl")
