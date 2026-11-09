from common import *

# s14 again, with the grouping rates at the FIXED threshold T = 0.66 (s16 random anchors, s17 coverage anchors)
# and a second ceiling. Kitaoka et al. (2026, arXiv 2610.24052): a per-agent monitor flags 6.7 percent (14 of 208)
# of distributed attacks at suspicion threshold 7.5; reading the team together flags 41 percent at the same
# threshold (their Section 5.1), and 31.7 percent (66 of 208) at threshold 8.5, the joint row at the same
# 2 percent false-positive budget (their Table C1). A model, not a measurement:
# E[detect] = P_joint * g + P_single * (1 - g); g = share of attack traces placed in the right group.
# Assumptions: linear in g; traces attributed independently; rates transfer; a misplaced trace adds no false group.

P_SINGLE = 14 / 208
CEIL = {"41% (same threshold)": 0.41, "31.7% (same FP budget)": 66 / 208}


def expect(g, pj):
    return pj * g + P_SINGLE * (1 - g)


rows = []
b = load("s16_out.pkl")
for k in sorted(b):
    v = np.array(b[k], float)
    rows.append((f"{k} random", np.nanmean(np.nan_to_num(v[:, 0] * v[:, 1])), np.nanmean(v[:, 2])))
c = load("s17_out.pkl")["B"]
for k in (5, 10):
    v = np.array(c[(k, "coverage")], float)
    rows.append((f"{k} coverage", np.nanmean(np.nan_to_num(v[:, 3] * v[:, 4])), np.nanmean(v[:, 2])))
a = load("s10b_out.pkl")[5]
rows.append(("signed chain, 5 signed", 0.0, a["top1"][0]))
print(f"{'anchors':>24} {'g refusal':>10} " + " ".join(f"{'E ' + n:>28}" for n in CEIL) + f" {'g top-1':>9} " + " ".join(f"{'E ' + n:>28}" for n in CEIL))
for name, gr, gt in rows:
    print(f"{name:>24} {gr:>10.3f} " + " ".join(f"{expect(gr, p):>28.3f}" for p in CEIL.values()) + f" {gt:>9.3f} " + " ".join(f"{expect(gt, p):>28.3f}" for p in CEIL.values()))
save(rows, "s18_out.pkl")
