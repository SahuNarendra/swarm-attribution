from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "figure.facecolor": "white", "savefig.dpi": 300, "savefig.bbox": "tight"})

# fig9: active anchoring (s17, the s13 runs plus the fixed threshold). Left: hidden pair rotated, top-1 accuracy
# on known files (no refusal), by anchor-selection rule. Right: signed chain, top-1 recognition by rule.
d = load("s17_out.pkl")
A, B = d["A"], d["B"]
rules = ("random", "coverage", "diversity", "least sure", "most sure")
col = {"random": "#8a8984", "coverage": "#2a78d6", "diversity": "#1baf7a", "least sure": "#eb6834", "most sure": "#c99a1e"}

fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.4))
K = sorted({k for k, _ in B})
for r in rules:
    V = [np.array(B[(k, r)], float)[:, 2] for k in K]
    m = [100 * np.nanmean(v) for v in V]
    lo = [100 * np.nanpercentile(v, 2.5) for v in V]; hi = [100 * np.nanpercentile(v, 97.5) for v in V]
    p = [np.nanmean(np.array(B[(k, r)], float)[:, 4]) for k in K]
    ax[0].plot(K, m, marker="o", ms=4, lw=2, color=col[r], label=r)
    if r in ("random", "coverage"):
        ax[0].fill_between(K, lo, hi, color=col[r], alpha=0.12, linewidth=0)
ax[0].set_xlabel("anchors per known lineage"); ax[0].set_ylabel("top-1 accuracy on known files (%)")
ax[0].set_title("Hidden lineage pairs (28): top-1", loc="left", fontsize=11, fontweight="bold", color="#0b0b0b")
ax[0].set_xticks(K); ax[0].set_ylim(0, 100); ax[0].legend(fontsize=8.5, frameon=False, loc="upper left", ncol=3); 

KA = sorted({k for k, _ in A})
for r in rules:
    ax[1].plot(KA, [100 * A[(k, r)]["top1"] for k in KA], marker="o", ms=4, lw=2, color=col[r], label=r)
ax[1].set_xlabel("signed anchors"); ax[1].set_ylabel("remaining signed files recognised (%)")
ax[1].set_title("Signed G236 chain: top-1", loc="left", fontsize=11, fontweight="bold", color="#0b0b0b")
ax[1].set_xticks(KA); ax[1].set_ylim(0, 105); ax[1].legend(fontsize=9, frameon=False, loc="lower right", ncol=2); 
plt.tight_layout(); plt.savefig(out("fig9_active_anchors.png"))
print("wrote fig9_active_anchors.png")
