from common import *
"""Pair features shared by s4_count and s4b_group."""
import time
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
t0 = time.time()
d = load("prep.pkl"); rows, X, ws = d["rows"], d["X"], d["ws"]
a = load("arts.pkl"); arts, df = a["arts"], a["df"]
N = len(rows)
Z = StandardScaler().fit_transform(X); Zn = normalize(Z)
W = normalize(TfidfVectorizer(token_pattern=r"\S+", min_df=3, sublinear_tf=True).fit_transform(ws)).tocsr()
types = np.array([r["ptype"] for r in rows]); lens = np.array([r["n_chars"] for r in rows], float)
rare = {x for x, n in df.items() if n <= 50}

def feats(P):
    I, J = P[:, 0], P[:, 1]
    D = np.abs(Z[I] - Z[J])
    eu = np.linalg.norm(Z[I] - Z[J], axis=1)[:, None]
    cs = (Zn[I] * Zn[J]).sum(1)[:, None]
    wc = np.asarray(W[I].multiply(W[J]).sum(1))
    A = np.zeros((len(P), 3))
    for k, (i, j) in enumerate(P):
        sh = arts[i] & arts[j]
        if sh:
            r = [x for x in sh if x in rare]
            A[k] = (len(r), sum(1 / math.log(df[x] + 1) for x in r), 1.0)
    meta = np.c_[(types[I] == types[J]).astype(float), np.abs(np.log(lens[I] / lens[J]))]
    return np.hstack([D, eu, cs, wc, A, meta])

