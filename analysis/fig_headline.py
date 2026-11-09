from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Headline figure: what anchors buy on the signed chain (s10, s10b), the anchor budget at the fixed T (s16),
# and how much an impersonator gets through (s12).
C1, C2, C3, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 11.5, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe7",
                     "axes.axisbelow": True, "figure.facecolor": "white", "savefig.dpi": 300, "savefig.bbox": "tight"})
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.05], "wspace": 0.32})

# A: signed chain
res = load("s10b_out.pkl"); K = sorted(res)
top = [100 * res[k]["top1"][0] for k in K]; top[0] = 100 * np.mean(load("s10_out.pkl")["res"]["top1"])
att = [100 * (res[k]["a5"][0] if res[k]["a5"][0] == res[k]["a5"][0] else 0) for k in K]
ax = axs[0]
ax.plot(K, top, color=C1, lw=2.4, marker="o", ms=5, label="best guess correct")
ax.plot(K, att, color=C3, lw=2.4, marker="o", ms=5, label="named with refusal\n(none wrongly)")
ax.axhline(12.5, color=GREY, lw=1, ls="--"); ax.text(10, 15.5, "chance", fontsize=9, color=GREY, ha="right")
for k, t in zip(K, top):
    if k in (0, 2, 5): ax.text(k - 0.25, t + 4, f"{t:.0f}", fontsize=10.5, color=C1, weight="bold", ha="right")
ax.set_xticks(K); ax.set_ylim(-3, 112); ax.set_xlabel("anchors (signed files given)"); ax.set_ylabel("% of the other signed files")
ax.legend(frameon=True, framealpha=1, edgecolor="white", fontsize=10.5, loc="center right", bbox_to_anchor=(1.0, 0.42))
ax.set_title("A. Signed chain: anchors recover it", loc="left", fontsize=12.5, weight="bold", color=INK)

# B: anchor budget at fixed T
b = load("s16_out.pkl"); A = sorted(b)
m = lambda k, i: 100 * np.nanmean([r[i] for r in b[k]])
ax = axs[1]; x = np.arange(len(A))
ax.plot(x, [m(k, 1) for k in A], color=C1, lw=2.4, marker="o", ms=5, label="precision")
ax.plot(x, [m(k, 0) for k in A], color=C3, lw=2.4, marker="o", ms=5, label="known files named")
ax.plot(x, [m(k, 4) for k in A], color=C2, lw=2.4, marker="o", ms=5, label="strangers wrongly named")
for i, k in enumerate(A):
    if k == 10:
        ax.text(i, m(k, 0) + 6, f"{m(k, 0):.0f}", ha="center", fontsize=10.5, color=C3, weight="bold")
        ax.text(i, m(k, 4) + 4, f"{m(k, 4):.1f}", ha="center", fontsize=10.5, color=C2, weight="bold")
        ax.text(i, m(k, 1) - 10, f"{m(k, 1):.0f}", ha="center", fontsize=10.5, color=C1, weight="bold")
ax.set_xticks(x); ax.set_xticklabels([str(k) for k in A]); ax.set_ylim(-3, 112)
ax.set_xlabel("verified files per known lineage"); ax.set_ylabel("% of files")
ax.legend(frameon=True, framealpha=1, edgecolor="white", fontsize=10.5, loc="center", bbox_to_anchor=(0.58, 0.64))
ax.set_title("B. The cost of anchors (T = 0.66)", loc="left", fontsize=12.5, weight="bold", color=INK)

# C: impersonation
s12 = load("s12_out.pkl")["res"]
f = lambda name, k: 100 * np.mean([r[k] for r in s12[name]])
keys = [("framed", "no\ndefence"), ("framed_conf", "refusal"), ("framed_conf_unflagged", "+ tamper\ncheck")]
ax = axs[2]; x = np.arange(3); w = 0.38
ax.bar(x - w / 2, [f("fixed_both", k) for k, _ in keys], w, color=C1, label="fixed attacker")
ax.bar(x + w / 2, [f("adaptive", k) for k, _ in keys], w, color=C2, label="adaptive attacker")
for i, (k, _) in enumerate(keys):
    ax.text(i - w / 2, f("fixed_both", k) + 2, f"{f('fixed_both', k):.0f}", ha="center", fontsize=10.5, color=INK)
    ax.text(i + w / 2, f("adaptive", k) + 2, f"{f('adaptive', k):.0f}", ha="center", fontsize=10.5, color=INK, weight="bold")
ax.set_xticks(x); ax.set_xticklabels([l for _, l in keys]); ax.set_ylim(0, 112); ax.set_ylabel("% of files framed on another lineage")
ax.legend(frameon=False, fontsize=10.5, loc="upper right")
ax.set_title("C. Impersonation gets through", loc="left", fontsize=12.5, weight="bold", color=INK)
fig.savefig(out("fig_headline.png"), facecolor="white")
print("A top", [round(t, 1) for t in top], "att", [round(a, 1) for a in att])
print("B", [(k, round(m(k, 0), 1), round(m(k, 1), 1), round(m(k, 4), 1)) for k in A])
print("C", [(k, round(f("fixed_both", k), 1), round(f("adaptive", k), 1)) for k, _ in keys])
