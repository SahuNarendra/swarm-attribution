from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Method overview: what goes in, what the pipeline does, what a responder gets back.
INK, INK2, C1, C3, C2, GREY = "#0b0b0b", "#52514e", "#2a78d6", "#1baf7a", "#eb6834", "#8a8984"
fig, ax = plt.subplots(figsize=(10, 2.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 2.6); ax.axis("off")
def box(x, w, title, body, col):
    ax.add_patch(FancyBboxPatch((x, 0.45), w, 1.8, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=col, lw=1.6))
    ax.text(x + w / 2, 2.0, title, ha="center", va="center", fontsize=11.5, weight="bold", color=INK)
    ax.text(x + w / 2, 1.12, body, ha="center", va="center", fontsize=9.8, color=INK2, linespacing=1.25)
def arrow(x0, x1):
    ax.annotate("", xy=(x1, 1.32), xytext=(x0, 1.32), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.4))
box(0.05, 1.75, "Artefacts", "files, commands\nand payloads\nleft behind", GREY)
arrow(1.82, 2.1)
box(2.1, 1.75, "Strip names", "controller names\ndeleted before\nany feature\nis computed", GREY)
arrow(3.87, 4.15)
box(4.15, 1.75, "Two lenses", "style: how code\nis written; work\nsignature: what\nit uses", C1)
arrow(5.92, 6.2)
box(6.2, 1.75, "Refusal", "a few verified\nfiles per lineage;\nrefuse below\nT = 0.66", C3)
arrow(7.97, 8.2)
box(8.2, 1.75, "Output", "lineage and\nconfidence, or\n\"unknown\"; files\nto verify", C2)
ax.text(5.0, 0.2, "Tested on: a held-out signed chain, hidden lineages, disguise, two impersonators, two other agent datasets.",
        ha="center", va="center", fontsize=10.5, color=INK2)
fig.savefig(out("fig_method.png"), dpi=300, bbox_inches="tight", facecolor="white")
print("wrote fig_method.png")
