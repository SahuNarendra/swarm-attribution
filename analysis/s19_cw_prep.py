"""s19: collusion.wiki (Von Arx et al. 2026): build units for attribution.
Unit = the text one labelled revision added (lines inserted or replaced by that edit, from its hunks), so a
page's earlier text by other names is not credited to this name. Units under 200 characters are dropped,
identical added texts that occur more than once are dropped, all known names are deleted from the text,
and only names with 10 or more units on 3 or more pages are kept. Structure only; no text is printed."""
from common import *
import json, collections
CW = os.path.join(ROOT, "data", "collusion-wiki")  # unzip full-wiki-logs.zip from https://collusion.wiki/explorer/download here (not committed)
from swarmattr.loader import Payload
from swarmattr.features import feature_matrix, CLUSTER_FEATURES

R = [json.loads(l) for l in open(os.path.join(CW, "revisions.jsonl"))]
names = sorted({r["label"] for r in R if r.get("label")}, key=len, reverse=True)
nameset = {n.lower() for n in names if len(n) >= 4}
_tok = re.compile(r"[A-Za-z0-9_]+")
strip = lambda t: _tok.sub(lambda m: "" if m.group().lower() in nameset else m.group(), t)

lab_all = [r for r in R if r.get("label")]
nets = collections.defaultdict(set)
for r in lab_all: nets[r["label"]].add(r["ip16"])
top = max(nets, key=lambda l: sum(1 for r in lab_all if r["label"] == l))
print("edits", len(R), "with a name", len(lab_all), "| most active name:", sum(1 for r in lab_all if r["label"] == top),
      "edits from", len(nets[top]), "/16 networks")
units = []
for r in R:
    if not r.get("label"): continue
    lines = (r["body"] or "").split("\n")
    added = []
    for h in r.get("hunks") or []:
        if h.get("op") in ("insert", "replace"):
            added += lines[h["b0"]:h["b1"]]
    t = "\n".join(added)
    if len(t) < 200: continue
    units.append(dict(label=r["label"], page=r["page_id"], time=r["time"], ip16=r["ip16"], text=strip(t)))
sha = collections.Counter(u["text"] for u in units)
units = [u for u in units if sha[u["text"]] == 1]
cnt = collections.Counter(u["label"] for u in units)
pages = collections.defaultdict(set)
for u in units: pages[u["label"]].add(u["page"])
keep = {l for l in cnt if cnt[l] >= 10 and len(pages[l]) >= 3}
units = [u for u in units if u["label"] in keep]
print("units", len(units), "names", len(keep), "pages", len({u["page"] for u in units}))
print("units per name: median", sorted(collections.Counter(u["label"] for u in units).values())[len(keep) // 2],
      "max", max(collections.Counter(u["label"] for u in units).values()))
X, fn = feature_matrix([Payload(rec_id=str(i), text=u["text"]) for i, u in enumerate(units)], CLUSTER_FEATURES)
save(dict(units=units, X=X, fn=fn), "s19_cw_prep.pkl")
