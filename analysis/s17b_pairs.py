from common import *

# Per-pair comparisons from s17_out.pkl, for every column of Setting B. Random is the mean of its 2 draws per pair.
B = load("s17_out.pkl")["B"]
cols = {0: "attributed, per-run threshold", 2: "top-1 without refusal", 3: "attributed, fixed T = 0.66"}
for j, name in cols.items():
    for k in (5, 10):
        r = np.array([x[j] for x in B[(k, "random")]]).reshape(-1, 2).mean(1)
        for rule in ("coverage", "diversity", "least sure", "most sure"):
            c = np.array([x[j] for x in B[(k, rule)]])
            print(f"{name}: k={k} {rule} beats random in {(c > r).sum()} of {len(c)} pairs (mean {c.mean():.3f} vs {r.mean():.3f})")
    c5 = np.array([x[j] for x in B[(5, "coverage")]]); r10 = np.array([x[j] for x in B[(10, "random")]]).reshape(-1, 2).mean(1)
    print(f"{name}: 5 coverage >= 10 random in {(c5 >= r10).sum()} of 28 pairs (mean {c5.mean():.3f} vs {r10.mean():.3f})")
