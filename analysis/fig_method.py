from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Method overview: what goes in, what the pipeline does, what a responder gets back.
INK, INK2, C1, C3, C2, GREY = "#0b0b0b", "#52514e", "#2a78d6", "#1baf7a", "#eb6834", "#8a8984"
fig, ax = plt.subplots(figsize=(10, 2.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 2.6); ax.axis("off")
def box(x, w, title, body, col):
    ax.add_patch(FancyBboxPatch((x, 0.55), w, 1.55, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=col, lw=1.6))
    ax.text(x + w / 2, 1.85, title, ha="center", va="center", fontsize=9.5, weight="bold", color=INK)
    ax.text(x + w / 2, 1.15, body, ha="center", va="center", fontsize=8, color=INK2, linespacing=1.35)
def arrow(x0, x1):
    ax.annotate("", xy=(x1, 1.32), xytext=(x0, 1.32), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.4))
box(0.05, 1.75, "Artefacts", "files, commands,\npayloads left behind;\nno names, no times", GREY)
arrow(1.82, 2.1)
box(2.1, 1.75, "Strip names", "controller names\ndeleted before any\nfeature is computed", GREY)
arrow(3.87, 4.15)
box(4.15, 1.75, "Two lenses", "style: how code\nis written; work\nsignature: what it uses", C1)
arrow(5.92, 6.2)
box(6.2, 1.75, "Anchors, refusal", "a few verified files\nper lineage; refuse\nbelow fixed T = 0.66", C3)
arrow(7.97, 8.2)
box(8.2, 1.75, "Responder gets", "lineage + confidence,\nor \"unknown\";\nwhich files to verify", C2)
ax.text(5.0, 0.2, "Tested against: a signed chain held out of training, hidden lineages, disguise, a fixed and an adaptive impersonator, and two other agent datasets.",
        ha="center", va="center", fontsize=8.5, color=INK2)
fig.savefig(out("fig_method.png"), dpi=300, bbox_inches="tight", facecolor="white")
print("wrote fig_method.png")
