from common import *
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

# Control for s10b: can file type alone, given the same k signed anchors, recognise the rest of the signed G236 chain?
# Same split and draws as s10b; the classifier sees only the payload type of each family.
d = load("prep.pkl"); rows = d["rows"]
L = [i for i, r in enumerate(rows) if r["label"]]
cnt = Counter(rows[i]["label"] for i in L); ctrls = sorted(c for c, n in cnt.items() if n >= 8)
L = [i for i in L if rows[i]["label"] in ctrls]
S = [i for i in L if rows[i]["signed"]]
TR = [i for i in L if i not in S]
T = np.array([[rows[i]["ptype"]] for i in range(len(rows))])
print("payload types of signed files:", Counter(rows[i]["ptype"] for i in S))
print("other lineages' files of that type:", Counter(rows[i]["label"] for i in TR if rows[i]["ptype"] in {rows[j]["ptype"] for j in S}))
rng = np.random.default_rng(0); REPS = 20
out = {}
for k in (0, 1, 2, 3, 5, 10):
    acc = []
    for rep in range(REPS if k else 3):
        A = list(rng.choice(S, size=k, replace=False)) if k else []
        te = [i for i in S if i not in A]
        idx = TR + A; y = np.array([rows[i]["label"] for i in idx])
        oh = OneHotEncoder(handle_unknown="ignore").fit(T[idx])
        m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(oh.transform(T[idx]), y)
        acc.append((m.predict(oh.transform(T[te])) == "G236").mean())
    a = np.array(acc); out[k] = (a.mean(), np.percentile(a, 2.5), np.percentile(a, 97.5))
    print(f"k={k:>2}: file-type-only top-1 G236 {a.mean():.3f} [{np.percentile(a, 2.5):.3f}, {np.percentile(a, 97.5):.3f}]")
save(out, "s24_out.pkl")
