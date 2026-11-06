from common import *
from disguise import *
rng = np.random.default_rng(0); ctrls = sorted(set(y)); out = {s: [] for s in (0.0, 0.75)}
for rep in range(10):
    unk = set(rng.choice(ctrls, 2, replace=False)); K = np.where(~np.isin(y, list(unk)))[0]; U = np.where(np.isin(y, list(unk)))[0]
    X0, W0 = feat[0.0]; sc, rf, tv, lr = models(X0[K], [W0[i] for i in K], y[K], rep)
    for s in out:
        Xs, Ws = feat[s]
        P = (rf.predict_proba(sc.transform(Xs[U])) + lr.predict_proba(tv.transform([Ws[i] for i in U]))) / 2
        out[s].append((P.max(1) >= 0.66).mean())
for s, v in out.items(): print(f"unknown-lineage files wrongly attributed at t=0.66, disguise {s}: {np.mean(v):.3f} +- {np.std(v):.3f}")
