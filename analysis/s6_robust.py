from common import *
from disguise import *
from sklearn.model_selection import StratifiedKFold
T = 0.66
res = {}
for mode in ("defender trained on clean", "defender trained on disguised too"):
    for s in LEVELS:
        acc = {"style": [], "work": [], "both": [], "attributed@0.66": [], "precision@0.66": []}
        for seed in range(3):
            for tr, te in StratifiedKFold(4, shuffle=True, random_state=seed).split(fam, y):
                X0, W0 = feat[0.0]; Xs, Ws = feat[s]
                if mode.startswith("defender trained on disguised") and s > 0:
                    Xtr = np.vstack([X0[tr], Xs[tr]]); Wtr = [W0[i] for i in tr] + [Ws[i] for i in tr]; ytr = np.r_[y[tr], y[tr]]
                else:
                    Xtr = X0[tr]; Wtr = [W0[i] for i in tr]; ytr = y[tr]
                sc, rf, tv, lr = models(Xtr, Wtr, ytr, seed)
                P1 = rf.predict_proba(sc.transform(Xs[te])); P2 = lr.predict_proba(tv.transform([Ws[i] for i in te]))
                c = rf.classes_
                acc["style"].append((c[P1.argmax(1)] == y[te]).mean()); acc["work"].append((c[P2.argmax(1)] == y[te]).mean())
                P = (P1 + P2) / 2; pred = c[P.argmax(1)]; ok = pred == y[te]; a = P.max(1) >= T
                acc["both"].append(ok.mean()); acc["attributed@0.66"].append(a.mean()); acc["precision@0.66"].append(ok[a].mean() if a.any() else np.nan)
        res[(mode, s)] = {k: (np.nanmean(v), np.nanstd(v)) for k, v in acc.items()}
    print(f"\n{mode.upper()} (disguise applied to test files only; majority baseline {max(Counter(y).values())/len(y):.3f})")
    print(f"{'disguise':<22}{'style':>14}{'work sig':>14}{'both':>14}{'attributed@t':>14}{'precision@t':>14}")
    for s in LEVELS:
        r = res[(mode, s)]
        print(f"{NAMES[s]:<22}" + "".join(f"{r[k][0]:>9.3f}+-{r[k][1]:.2f}" for k in ("style", "work", "both", "attributed@0.66", "precision@0.66")), flush=True)
save(res, "s6_out.pkl")
