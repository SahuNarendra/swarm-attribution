from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
C1, C2, C3, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "figure.facecolor": "white", "savefig.dpi": 300, "savefig.bbox": "tight"})

# Fig 7: the signed chain, recognised as anchors are added (s10b)
res = load("s10b_out.pkl")
K = sorted(res)
top = [100 * res[k]["top1"][0] for k in K]
top[0] = 100 * np.mean(load("s10_out.pkl")["res"]["top1"])  # k = 0: the 10-seed held-out test of s10 (3.3 percent), as in the text and Table A6
lo = [100 * res[k]["top1"][1] for k in K]; hi = [100 * res[k]["top1"][2] for k in K]
att = [100 * (res[k]["a5"][0] if res[k]["a5"][0] == res[k]["a5"][0] else 0) for k in K]
wrong = [100 * res[k]["w5"][0] for k in K]
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.fill_between(K, lo, hi, color=C1, alpha=0.15, lw=0)
ax.plot(K, top, color=C1, lw=2, marker="o", ms=4, label="signed files recognised as G236 (top-1)")
ax.plot(K, att, color=C3, lw=2.5, marker="o", ms=5, zorder=5, label="attributed with refusal (5% stranger setting)")
ax.plot(K, wrong, color=C2, lw=2, marker="o", ms=4, label="wrongly and confidently attributed")
for k, t in zip(K, top): ax.text(k + 0.3, t + 6, f"{t:.1f}" if t < 10 else f"{t:.0f}", ha="center", fontsize=8, color=INK)
ax.axhline(12.5, color=GREY, lw=1, ls="--"); ax.text(10.3, 14, "chance (1 of 8)", fontsize=8, color=GREY, ha="right")
ax.set_xticks(K); ax.set_xlabel("signed files given as anchors"); ax.set_ylabel("% of the remaining signed files"); ax.set_ylim(-3, 110)
ax.legend(frameon=False, fontsize=8, loc="center right")
ax.set_title("Signed G236 chain, never in training", loc="left", color=INK, fontsize=11, fontweight="bold")
fig.savefig(out("fig7_signed_anchors.png"))

# Fig 8: fixed vs adaptive impersonation (s9 both vs s12 adaptive)
s12 = load("s12_out.pkl")["res"]
f = lambda name, k: 100 * np.mean([r[k] for r in s12[name]])
keys = [("framed", "blamed on B\nno refusal"), ("framed_conf", "confidently blamed on B\nwith refusal"), ("framed_conf_unflagged", "blamed, with refusal,\nmissed by the tamper check")]
fig, ax = plt.subplots(figsize=(6.6, 3.4))
x = np.arange(3); w = 0.36
ax.bar(x - w / 2, [f("fixed_both", k) for k, _ in keys], w, color=C1, label="fixed attacker (both moves)")
ax.bar(x + w / 2, [f("adaptive", k) for k, _ in keys], w, color=C2, label="adaptive attacker (surrogate, 16 tries)")
for i, (k, _) in enumerate(keys):
    ax.text(i - w / 2, f("fixed_both", k) + 1.5, f"{f('fixed_both', k):.0f}", ha="center", fontsize=8, color=INK)
    ax.text(i + w / 2, f("adaptive", k) + 1.5, f"{f('adaptive', k):.0f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in keys], fontsize=8.5); ax.set_ylim(0, 110); ax.set_ylabel("% of A's files")
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("Impersonation: refusal and the tamper check against two attackers", loc="left", color=INK, fontsize=11, fontweight="bold")
fig.savefig(out("fig8_adaptive.png"))
print("fig7, fig8 written")
