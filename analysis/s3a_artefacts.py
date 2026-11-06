from common import *
from collections import Counter
from swarmattr.loader import load_records, build_families, Payload
from swarmattr.graph import shared_artefacts
recs = load_records(DATA)
big = [f for f in build_families(recs) if f.n_chars >= 200]
arts = []
for f in big:
    s = shared_artefacts(Payload(rec_id=f.rec_id, text=strip_names(f.text)), kinds=("marker", "file", "host", "paste"))
    arts.append({a for a in s if "runtime_identifier" not in a})
df = Counter(a for s in arts for a in s)
save(dict(arts=arts, df=df), "arts.pkl")
print("families with any artefact:", sum(1 for s in arts if s), "distinct artefacts:", len(df))
print("artefacts in 2..50 families:", sum(1 for a, n in df.items() if 2 <= n <= 50))
print("top 10 most common:", df.most_common(10))
