from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
C1, C2, C3, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "figure.facecolor": "white", "savefig.dpi": 300, "savefig.bbox": "tight"})

# Fig 5: impersonation
res = load("s9_out.pkl")
moves = ["none", "splice", "vocab", "both"]
f = lambda m, k: 100 * np.mean([r[k] for r in res[m]])
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.2), gridspec_kw={"wspace": 0.3})
x = np.arange(4); w = 0.38
ax = axs[0]
ax.bar(x - w / 2, [f(m, "framed") for m in moves], w, color=C2, label="blamed on B")
ax.bar(x + w / 2, [f(m, "caught") for m in moves], w, color=C1, label="caught as A")
for i, m in enumerate(moves):
    ax.text(i - w / 2, f(m, "framed") + 1.5, f"{f(m,'framed'):.0f}", ha="center", fontsize=8, color=INK)
    ax.text(i + w / 2, f(m, "caught") + 1.5, f"{f(m,'caught'):.0f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(x); ax.set_xticklabels(moves); ax.set_ylim(0, 100); ax.set_ylabel("% of A's files"); ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("No refusal: top-1 attribution", loc="left", color=INK, fontsize=11, fontweight="bold")
ax = axs[1]
ax.bar(x - w / 2, [f(m, "framed_conf") for m in moves], w, color=C2, label="confidently blamed on B")
ax.bar(x + w / 2, [f(m, "caught_conf") for m in moves], w, color=C1, label="confidently caught as A")
for i, m in enumerate(moves):
    ax.text(i - w / 2, f(m, "framed_conf") + 1.5, f"{f(m,'framed_conf'):.0f}", ha="center", fontsize=8, color=INK)
    ax.text(i + w / 2, f(m, "caught_conf") + 1.5, f"{f(m,'caught_conf'):.0f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(x); ax.set_xticklabels(moves); ax.set_ylim(0, 100); ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("With refusal (threshold 0.66)", loc="left", color=INK, fontsize=11, fontweight="bold")
fig.savefig(out("fig5_impersonation.png"))

# Fig 6: anchor budget
res = load("s8_out.pkl")
K = sorted(res); m = {k: np.nanmean(np.array(res[k], float), 0) for k in K}
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.plot(K, [100 * m[k][1] for k in K], color=C1, lw=2, marker="o", ms=4)
ax.plot(K, [100 * m[k][0] for k in K], color=C3, lw=2, marker="o", ms=4)
ax.plot(K, [100 * m[k][2] for k in K], color=GREY, lw=1.5, ls=":", marker="o", ms=3)
ax.text(41, 100 * m[40][1] - 1, "precision of attributions", fontsize=8, color=INK, va="center")
ax.text(41, 100 * m[40][0], "share of files attributed", fontsize=8, color=INK, va="center")
ax.text(41, 100 * m[40][2] + 1, "top-1 without refusal", fontsize=8, color=INK, va="center")
ax.set_xlabel("anchored files per lineage"); ax.set_ylabel("%"); ax.set_xlim(0, 62); ax.set_ylim(0, 105); ax.set_xticks(K)
ax.set_title("How many anchors a responder needs (5% stranger error)", loc="left", color=INK, fontsize=11, fontweight="bold")
fig.savefig(out("fig6_anchor_budget.png"))
print("ok")
