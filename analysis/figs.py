from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
C1, C2, C3, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "figure.facecolor": "white", "savefig.dpi": 300, "savefig.bbox": "tight"})
def finish(ax, title):
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")

# Fig 1: two lenses, two datasets
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.2), gridspec_kw={"width_ratios": [3, 2], "wspace": 0.35})
for ax, (name, labels, vals, cols, base) in zip(axs, [
        ("Swarm Traces: 10 controllers", ["Style", "Work signature", "Both"], [75.0, 69.7, 80.7], [C1, C2, C3], 30.4),
        ("AI Village: 133 instances, one model", ["Style", "Work signature"], [2.2, 30.1], [C1, C2], 1.4)]):
    xs = range(len(vals)); ax.bar(xs, vals, color=cols, width=0.6)
    for x in xs: ax.text(x, vals[x] + 1.5, f"{vals[x]:.1f}%", ha="center", color=INK, fontsize=9)
    ax.axhline(base, color=GREY, ls="--", lw=1.2); ax.text(len(vals) - 0.62, base + (9 if base < 5 else 0), f"majority\n{base}%", color=INK2, fontsize=8, va="center", ha="left", clip_on=False)
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels); ax.set_ylim(0, 100)
    if ax is axs[0]: ax.set_ylabel("top-1 accuracy (%)")
    finish(ax, name)
fig.savefig(out("fig1_two_lenses.png"))

# Fig 2: where identity lives (AI Village)
fig, ax = plt.subplots(figsize=(5.2, 3.0))
labs = ["Same session,\nother half", "Same agent,\nother session", "Different\nagent"]; v = [0.41, 0.24, 0.21]
ax.bar(range(3), v, color=[C1, C2, GREY], width=0.6)
for i, x in enumerate(v): ax.text(i, x + 0.01, f"{x:.2f}", ha="center", color=INK, fontsize=9)
ax.text(0.12, 0.46, "AUC 0.86 vs different agent", ha="center", color=INK2, fontsize=8); ax.text(1, 0.285, "AUC 0.56 vs different agent", ha="center", color=INK2, fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels(labs); ax.set_ylabel("mean cosine similarity\n(command vocabulary)"); ax.set_ylim(0, 0.62)
finish(ax, "AI Village, 133 instances of one model: who matches whom")
fig.savefig(out("fig2_where_identity_lives.png"))

# Fig 3: refusal trade-off (open set)
t = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
att = [100, 95.4, 82.2, 64.8, 46.3, 33.1, 20.4]; prec = [85.6, 87.9, 93.2, 98.0, 99.7, 100, 100]; unk = [100, 75.4, 36.7, 18.2, 9.3, 1.5, 0.0]
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(t, prec, color=C1, lw=2, marker="o", ms=4); ax.plot(t, att, color=C3, lw=2, marker="o", ms=4); ax.plot(t, unk, color=C2, lw=2, marker="o", ms=4)
ax.text(0.81, prec[-1] - 6, "precision on known lineages", color=INK, fontsize=8)
ax.text(0.83, att[-1] - 2, "known files attributed", color=INK, fontsize=8)
ax.text(0.81, unk[-1] + 2, "strangers wrongly attributed", color=INK, fontsize=8)
ax.axvline(0.66, color=GREY, ls="--", lw=1); ax.text(0.68, 52, "5% stranger\nerror", color=INK2, fontsize=8)
ax.set_xlim(0, 0.85); ax.set_ylim(0, 105); ax.set_xlabel("confidence threshold"); ax.set_ylabel("%")
finish(ax, "Anchored attribution with refusal (2 lineages hidden, 20 reps)")
fig.savefig(out("fig3_refusal_tradeoff.png"))

# Fig 4: robustness
lv = ["none", "rename\nidentifiers", "+ reindent", "+ strip\ncomments", "+ strip names\n(already removed)"]
st = [78.5, 68.1, 64.1, 63.1, 63.1]; wk = [73.8, 53.0, 53.0, 53.0, 53.0]; bo = [82.8, 72.9, 72.2, 72.0, 72.0]; adv = [82.8, 81.7, 81.8, 81.8, 81.8]
fig, ax = plt.subplots(figsize=(6.4, 3.4)); x = range(5)
ax.plot(x, bo, color=C3, lw=2, marker="o", ms=4); ax.plot(x, st, color=C1, lw=2, marker="o", ms=4); ax.plot(x, wk, color=C2, lw=2, marker="o", ms=4)
ax.plot(x, adv, color=C3, lw=1.5, ls=":", marker="o", ms=3)
ax.axhline(31.2, color=GREY, ls="--", lw=1.2)
for y, lab in ((adv[-1] + 1, "both, defender trained on disguise"), (bo[-1] + 1, "both, defender trained on clean"), (st[-1] + 1, "style"), (wk[-1] + 1, "work signature"), (32.5, "majority baseline")):
    ax.text(4.1, y, lab, color=INK, fontsize=8, va="bottom")
ax.set_xticks(list(x)); ax.set_xticklabels(lv, fontsize=8); ax.set_xlim(-0.2, 5.9); ax.set_ylim(0, 100); ax.set_ylabel("top-1 accuracy (%)")
finish(ax, "Attribution under disguise (8 controllers, 382 families)")
fig.savefig(out("fig4_robustness.png"))
print("ok")
