"""s22: collusion.wiki, do names that worked together write alike? "Worked together" = the two names edited at least one
common wiki page. For each name, a vocabulary profile is built from the text it added (s19 units), using ONLY pages the
pair did not share, so a shared page cannot carry the signal. Compares co-editing pairs with pairs that never shared a page:
mean cosine and AUC; then, as a responder would use it, for each name the top-5 most similar other names and how many of
those are real co-editors (precision at 5) against the rate expected by chance."""
from common import *
import collections, warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score

U = load("s19_cw_prep.pkl")["units"]
txt = [u["text"] for u in U]; lab = np.array([u["label"] for u in U]); pg = np.array([u["page"] for u in U])
V = normalize(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000).fit_transform(txt))
names = sorted(set(lab))
pages = {n: set(pg[lab == n]) for n in names}
idx = {n: np.where(lab == n)[0] for n in names}
def prof(n, excl):
    k = [i for i in idx[n] if pg[i] not in excl]
    if len(k) < 3: return None
    return normalize(np.asarray(V[k].mean(0)))[0]
co, nco = [], []
for a in range(len(names)):
    for b in range(a + 1, len(names)):
        A, B = names[a], names[b]
        shared = pages[A] & pages[B]
        pa, pb = prof(A, shared), prof(B, shared)
        if pa is None or pb is None: continue
        (co if shared else nco).append(float(pa @ pb))
auc = roc_auc_score(np.r_[np.ones(len(co)), np.zeros(len(nco))], np.r_[co, nco])
print(f"names {len(names)}; co-editing pairs {len(co)}, never co-edited {len(nco)}")
print(f"cosine (shared pages excluded): co-editors {np.mean(co):.3f} vs others {np.mean(nco):.3f}; AUC {auc:.3f}")
# precision at 5 with whole-name profiles minus nothing (responder view), excluding co-edited pages from both sides is
# not possible for a ranking, so profiles here use all units and we report it as the optimistic view
P = normalize(np.vstack([np.asarray(V[idx[n]].mean(0)) for n in names])); C = P @ P.T; np.fill_diagonal(C, -1)
hits, base = [], []
for i, n in enumerate(names):
    partners = {m for m in names if m != n and pages[n] & pages[m]}
    if not partners: continue
    top = np.argsort(-C[i])[:5]
    hits.append(np.mean([names[j] in partners for j in top])); base.append(len(partners) / (len(names) - 1))
print(f"top-5 most similar names that are real co-editors: {np.mean(hits):.3f} (chance {np.mean(base):.3f}; {len(hits)} names with any co-editor)")
# control for time: keep only pairs whose activity windows overlap (both names active in a common span)
from datetime import datetime
ts = np.array([datetime.fromisoformat(u["time"].replace("Z", "+00:00")).timestamp() for u in U])
win = {n: (ts[idx[n]].min(), ts[idx[n]].max()) for n in names}
co2, nco2 = [], []
for a in range(len(names)):
    for b in range(a + 1, len(names)):
        A, B = names[a], names[b]
        if min(win[A][1], win[B][1]) <= max(win[A][0], win[B][0]): continue
        shared = pages[A] & pages[B]
        pa, pb = prof(A, shared), prof(B, shared)
        if pa is None or pb is None: continue
        (co2 if shared else nco2).append(float(pa @ pb))
auc2 = roc_auc_score(np.r_[np.ones(len(co2)), np.zeros(len(nco2))], np.r_[co2, nco2])
print(f"time-matched (activity windows overlap): co-editors {np.mean(co2):.3f} ({len(co2)}) vs others {np.mean(nco2):.3f} ({len(nco2)}); AUC {auc2:.3f}")
save(dict(auc2=auc2, auc=auc, co=np.mean(co), nco=np.mean(nco), p5=np.mean(hits), p5_base=np.mean(base)), "s22_out.pkl")
