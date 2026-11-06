from common import *
from collections import Counter
from swarmattr.loader import load_records, build_families, label_controllers, is_signed_command, Payload
from swarmattr.features import feature_matrix, CLUSTER_FEATURES, work_signature

recs = load_records(DATA)
fams = build_families(recs)
big = [f for f in fams if f.n_chars >= 200]
print("records", len(recs), "families", len(fams), ">=200", len(big), flush=True)



signed_ids = {r.rec_id for r in recs if is_signed_command(r)}
rows = []
for f in big:
    labs = label_controllers(f.text, f.tags)
    sig = f.root.rec_id in signed_ids or any(c.rec_id in signed_ids for c in f.children)
    rows.append(dict(rec_id=f.rec_id, n_chars=f.n_chars, ptype=ptype(f.text),
                     label=labs[0] if len(labs) == 1 else None, n_labels=len(labs), signed=sig))
pl = [Payload(rec_id=f.rec_id, text=strip_names(f.text)) for f in big]
X, names = feature_matrix(pl, CLUSTER_FEATURES)
ws = [work_signature(p) for p in pl]
save(dict(rows=rows, X=X, names=names, ws=ws), "prep.pkl")
lab = Counter(r["label"] for r in rows if r["label"])
print("single-label families:", sum(lab.values()), lab.most_common())
print("signed families:", sum(r["signed"] for r in rows), "labels on signed:", Counter(r["label"] for r in rows if r["signed"]))
print("types:", Counter(r["ptype"] for r in rows).most_common())
print("empty work signatures:", sum(1 for w in ws if not w.strip()))
