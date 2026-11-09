"""Figure: what the artefacts of a one-model swarm can and cannot tell, by level.
Every number is taken from a script output in this repo (named in the right column) or from the
declared prior work on AI Village (5 Nov). Hand-laid figure, no data loaded.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis", "out")

rows = [
    ("Model", "which model wrote it",
     "one answer for ~95% of the swarm", "model fingerprinting (prior art)", "recoverable, uninformative", 0.25),
    ("Lineage", "which controller",
     "80.7% top-1 vs 30.4% baseline when seen;\nhidden lineage: 0 of 27 attributed at 0 anchors,\n97% top-1 at 5 signed anchors, 0 confident errors",
     "this paper, 4.1 to 4.3", "recoverable with anchors", 0.85),
    ("Trajectory", "same run as another artefact",
     "halves of one session match, AUC 0.86;\nsame agent after reset, AUC 0.56",
     "AI Village control (prior work)", "recoverable within a run", 0.6),
    ("Instance", "which copy of the model",
     "style at chance (2.2%), work signature 30% top-1\nacross 133 instances; does not survive a reset", "AI Village control (prior work)", "weakly recoverable", 0.2),
]

fig, ax = plt.subplots(figsize=(10.5, 5.2))
ax.set_xlim(0, 10.5); ax.set_ylim(0, 5.2); ax.axis("off")
cols = {"recoverable, uninformative": "#b0b0b0", "recoverable with anchors": "#2b6ca3",
        "recoverable within a run": "#5b9bd5", "weakly recoverable": "#c0504d"}
y0 = 4.4
for i, (lvl, q, ev, src, verdict, fill) in enumerate(rows):
    y = y0 - i * 1.1
    ax.add_patch(FancyBboxPatch((0.15, y - 0.42), 1.5, 0.84, boxstyle="round,pad=0.02", fc="#f2f2f2", ec="#888"))
    ax.text(0.9, y + 0.12, lvl, ha="center", va="center", fontsize=12, weight="bold")
    ax.text(0.9, y - 0.2, q, ha="center", va="center", fontsize=9.5, color="#444")
    ax.add_patch(FancyBboxPatch((1.9, y - 0.14), 2.6, 0.28, boxstyle="round,pad=0.02", fc="#eeeeee", ec="none"))
    ax.add_patch(FancyBboxPatch((1.9, y - 0.14), 2.6 * fill, 0.28, boxstyle="round,pad=0.02", fc=cols[verdict], ec="none"))
    ax.text(3.2, y - 0.32, verdict, ha="center", va="top", fontsize=10.5, color=cols[verdict], weight="bold")
    ax.text(4.75, y, ev, ha="left", va="center", fontsize=11)
    ax.text(10.35, y, src, ha="right", va="center", fontsize=9.5, color="#666", style="italic")
ax.text(0.9, 5.0, "level asked", ha="center", fontsize=11, color="#666")
ax.text(3.2, 5.0, "how much the artefacts give back", ha="center", fontsize=11, color="#666")
ax.text(4.75, 5.0, "evidence", ha="left", fontsize=11, color="#666")
ax.text(10.35, 5.0, "source", ha="right", fontsize=11, color="#666")
ax.plot([0.15, 10.35], [4.88, 4.88], color="#999", lw=0.8)
ax.text(5.25, 0.05, "Bar length is a qualitative summary of the evidence column. Anchors: verified files with a known author.", ha="center", fontsize=9.5, color="#555")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig0_ladder.png"), dpi=200)
print("wrote fig0_ladder.png")
