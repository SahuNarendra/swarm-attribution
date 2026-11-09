from common import *

# A model, not a measurement. Kitaoka et al. (2026, arXiv 2610.24052) report that a per-trajectory monitor
# flags 7 percent of successful distributed attacks (14 of 208) and 41 percent when the team's trajectories
# are read together, with the grouping given by the environment. If the grouping instead has to be recovered
# from artefacts, a trace that is attributed to the right lineage is read jointly and one that is refused or
# misattributed is read alone. Expected detection = P(joint) * g + P(single) * (1 - g), where g is the share of
# attack traces placed in the right group. Assumptions: linear in g; traces attributed independently; the
# monitor's detection rates transfer from MiniLab to this setting; a misattributed trace adds no false group.
# g is read from this repo's outputs: s8b (hidden pair rotated, 5 percent of hidden-lineage files attributed)
# and s10b (signed chain). Two readings: with refusal (attributed and correct) and top-1 without refusal.

P_SINGLE, P_JOINT = 14 / 208, 0.41


def expect(g):
    return P_JOINT * g + P_SINGLE * (1 - g)


print(f"Kitaoka et al.: per-trajectory {P_SINGLE:.3f}, joint with known grouping {P_JOINT:.2f}\n")
print("s8b, known lineages, k anchors each (mean over 28 hidden pairs x 2 draws)")
print(f"{'anchors':>8} {'g refusal':>10} {'E[detect]':>10} {'g top-1':>9} {'E[detect]':>10}")
b = load("s8b_out.pkl")
rows = []
for k in sorted(b):
    v = np.array(b[k], float)
    g_ref = np.nanmean(v[:, 0] * v[:, 1])   # attributed share x precision = correctly grouped share
    g_top = np.nanmean(v[:, 2])
    rows.append((k, g_ref, expect(g_ref), g_top, expect(g_top)))
    print(f"{k:>8} {g_ref:>10.3f} {expect(g_ref):>10.3f} {g_top:>9.3f} {expect(g_top):>10.3f}")
print("\ns10b, signed G236 chain, k signed anchors")
print(f"{'anchors':>8} {'g refusal':>10} {'E[detect]':>10} {'g top-1':>9} {'E[detect]':>10}")
a = load("s10b_out.pkl")
for k in sorted(a):
    m = a[k]
    g_ref = m["a5"][0] * (m["p5"][0] if m["p5"][0] == m["p5"][0] else 0.0)
    g_top = m["top1"][0]
    rows.append(("G236-" + str(k), g_ref, expect(g_ref), g_top, expect(g_top)))
    print(f"{k:>8} {g_ref:>10.3f} {expect(g_ref):>10.3f} {g_top:>9.3f} {expect(g_top):>10.3f}")
print("\nReading: with no anchors g = 0 and the joint monitor is worth nothing more than the single one. "
      "Anchors move g, and E[detect] with it, toward the 0.41 ceiling that assumes perfect grouping.")
save(rows, "s14_out.pkl")
