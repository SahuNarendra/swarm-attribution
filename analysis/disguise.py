from common import *
"""Disguised feature sets and the two-lens models, shared by s6_robust and s6b_unknown."""
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from swarmattr.loader import load_records, build_families, label_controllers, Payload
from swarmattr.features import feature_matrix, CLUSTER_FEATURES, work_signature
from swarmattr.adversary import perturb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

recs = load_records(DATA)
fam, y = [], []
for f in build_families(recs):
    if f.n_chars < 200: continue
    labs = label_controllers(f.text, f.tags)
    if len(labs) == 1: fam.append(Payload(rec_id=f.rec_id, text=strip_names(f.text))); y.append(labs[0])
cnt = Counter(y); keep = [k for k, c in enumerate(y) if cnt[c] >= 8]
fam = [fam[k] for k in keep]; y = np.array([y[k] for k in keep])
print("families", len(fam), "controllers", len(set(y)), flush=True)
LEVELS = [0.0, 0.1, 0.25, 0.5, 0.75]
NAMES = {0.0: "none", 0.1: "rename identifiers", 0.25: "+ reindent", 0.5: "+ strip comments", 0.75: "+ strip names"}
feat = {}
for s in LEVELS:
    pl = [perturb(p, s, seed=1) if s > 0 else p for p in fam]
    X, _ = feature_matrix(pl, CLUSTER_FEATURES); feat[s] = (X, [work_signature(p) for p in pl])
print("features done", flush=True)

def models(Xtr, Wtr, ytr, seed):
    sc = StandardScaler().fit(Xtr); rf = RandomForestClassifier(300, random_state=seed, class_weight="balanced", n_jobs=2).fit(sc.transform(Xtr), ytr)
    tv = TfidfVectorizer(token_pattern=r"\S+", min_df=2, sublinear_tf=True).fit(Wtr)
    lr = LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced").fit(tv.transform(Wtr), ytr)
    return sc, rf, tv, lr

