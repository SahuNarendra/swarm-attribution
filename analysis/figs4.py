from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fig9: active anchoring (s17, the s13 runs plus the fixed threshold). Left: hidden pair rotated, top-1 accuracy
# on known files (no refusal), by anchor-selection rule. Right: signed chain, top-1 recognition by rule.
d = load("s17_out.pkl")
A, B = d["A"], d["B"]
rules = ("random", "coverage", "diversity", "least sure", "most sure")
col = {"random": "#888888", "coverage": "#2b6ca3", "diversity": "#5b9bd5", "least sure": "#c0504d", "most sure": "#e0a0a0"}

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
K = sorted({k for k, _ in B})
for r in rules:
    V = [np.array(B[(k, r)], float)[:, 2] for k in K]
    m = [np.nanmean(v) for v in V]
    lo = [np.nanpercentile(v, 2.5) for v in V]; hi = [np.nanpercentile(v, 97.5) for v in V]
    p = [np.nanmean(np.array(B[(k, r)], float)[:, 4]) for k in K]
    ax[0].plot(K, m, marker="o", color=col[r], label=r)
    if r in ("random", "coverage"):
        ax[0].fill_between(K, lo, hi, color=col[r], alpha=0.12, linewidth=0)
ax[0].set_xlabel("anchors per known lineage"); ax[0].set_ylabel("top-1 accuracy on known files")
ax[0].set_title("Hidden pair rotated, top-1 (bands: 2.5 to 97.5 pct over pairs)", fontsize=10)
ax[0].set_xticks(K); ax[0].set_ylim(0, 1.0); ax[0].legend(fontsize=9, frameon=False, loc="lower center", ncol=3); ax[0].grid(alpha=0.3)

KA = sorted({k for k, _ in A})
for r in rules:
    ax[1].plot(KA, [A[(k, r)]["top1"] for k in KA], marker="o", color=col[r], label=r)
ax[1].set_xlabel("signed anchors"); ax[1].set_ylabel("top-1 recognition of the other signed files")
ax[1].set_title("Signed chain: how anchors are chosen", fontsize=10)
ax[1].set_xticks(KA); ax[1].set_ylim(0, 1.05); ax[1].legend(fontsize=9, frameon=False, loc="lower right", ncol=2); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(out("fig9_active_anchors.png"), dpi=200)
print("wrote fig9_active_anchors.png")
