"""s21: AI Village swarm village, team recovery on real agents of one model (glm-5.3-flash), with exact team labels.
Data (gated, aidigestorg/ai-village, swarm village; not committed): computer_use_turns.jsonl.gz, computer_use_sessions.jsonl.gz
and agents.jsonl.gz in data/aiv-swarm/. Unit = one session's bash commands of 120+ characters, joined (a session document);
agent and team come from the dataset (team = the colour in the agent name). Work-signature lens only (TF-IDF of command
tokens, logistic regression), because style is at chance on short bash commands (prior work, 5 Nov).
(A) Which team does an agent never seen in training belong to? Splits grouped by AGENT.
(B) Agent profiles (mean of its session vectors): cosine same team vs different team, AUC.
(C) Blind grouping: agglomerative clustering of agent profiles into k groups, purity and adjusted Rand against teams.
(D) Anchors: k agents per team labelled, every other agent assigned to a team (top-1 and with refusal at a threshold
    set on separate repetitions so 5 percent of agents from hidden teams are wrongly assigned)."""
from common import *
import gzip, json, collections, warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

D = os.path.join(ROOT, "data", "aiv-swarm")
COLOURS = ["amber", "blue", "coral", "green", "jade", "onyx", "pearl", "red", "ruby", "sage", "teal", "violet"]
agents = {}
for l in gzip.open(os.path.join(D, "agents.jsonl.gz"), "rt"):
    r = json.loads(l); agents[r["id"]] = r.get("name") or ""
def team_of(name):
    n = name.lower()
    hits = [c for c in COLOURS if re.search(r"(?<![a-z])" + c + r"(?![a-z])", n)]
    return hits[0] if len(hits) == 1 else None
s2a = {}
for l in gzip.open(os.path.join(D, "computer_use_sessions.jsonl.gz"), "rt"):
    r = json.loads(l); s2a[r["id"]] = r["agent_id"]
docs = collections.defaultdict(list)
for l in gzip.open(os.path.join(D, "computer_use_turns.jsonl.gz"), "rt", encoding="utf-8", errors="replace"):
    if not l.strip(): continue
    try: r = json.loads(l)
    except Exception: continue
    s = r.get("session_id"); a = s2a.get(s)
    if not a: continue
    am = r.get("agent_messages") or {}
    calls = am.get("tool_calls") if isinstance(am, dict) else None
    for tc in calls or []:
        f = tc.get("function") or {}
        if f.get("name") != "bash": continue
        try: t = json.loads(f.get("arguments") or "{}").get("command")
        except Exception: continue
        if isinstance(t, str) and len(t) >= 120: docs[s].append(t)
tok = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{1,}")
names_lc = {n.lower() for n in agents.values() if n} | set(COLOURS)
def clean(t):  # drop agent names and team colours so the label cannot be read off the text
    return " ".join(w for w in tok.findall(t) if w.lower() not in names_lc and not any(c in w.lower() for c in COLOURS))
S = [s for s in docs if len(docs[s]) >= 3]
ag = np.array([s2a[s] for s in S]); team = np.array([team_of(agents.get(a, "")) for a in ag], dtype=object)
keep = np.array([t is not None for t in team]); S = [s for s, k in zip(S, keep) if k]; ag = ag[keep]; team = team[keep].astype(str)
text = [clean("\n".join(docs[s])) for s in S]
print("sessions", len(S), "agents", len(set(ag)), "teams", len(set(team)), "agents per team", sorted(collections.Counter({a: t for a, t in zip(ag, team)}.values()).items()), flush=True)

# (A) team of an unseen agent
acc, base = [], []
for seed in range(5):
    for tr, te in StratifiedGroupKFold(4, shuffle=True, random_state=seed).split(np.zeros(len(S)), team, ag):
        tv = TfidfVectorizer(min_df=2, sublinear_tf=True).fit([text[i] for i in tr])
        lr = LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced").fit(tv.transform([text[i] for i in tr]), team[tr])
        P = lr.predict_proba(tv.transform([text[i] for i in te]))
        # agent-level vote: average the session probabilities of each test agent
        for a in set(ag[te]):
            m = ag[te] == a
            acc.append(lr.classes_[P[m].mean(0).argmax()] == team[te][m][0])
        base.append((team[te] == collections.Counter(team[tr]).most_common(1)[0][0]).mean())
print(f"\n(A) TEAM OF AN UNSEEN AGENT (agent-level, sessions averaged; splits grouped by agent): {np.mean(acc):.3f} "
      f"over {len(acc)} agent predictions; chance {1/len(set(team)):.3f}, majority {np.mean(base):.3f}")

# (B) agent profiles
tv = TfidfVectorizer(min_df=2, sublinear_tf=True).fit(text); V = tv.transform(text)
A = sorted(set(ag)); at = {a: team[ag == a][0] for a in A}
Pf = normalize(np.vstack([np.asarray(V[ag == a].mean(0)) for a in A]))
C = Pf @ Pf.T
same, diff = [], []
for i in range(len(A)):
    for j in range(i + 1, len(A)):
        (same if at[A[i]] == at[A[j]] else diff).append(C[i, j])
auc = roc_auc_score(np.r_[np.ones(len(same)), np.zeros(len(diff))], np.r_[same, diff])
print(f"\n(B) AGENT PROFILES: cosine same team {np.mean(same):.3f} ({len(same)} pairs) vs different teams {np.mean(diff):.3f} ({len(diff)}); AUC {auc:.3f}")

# (C) blind grouping
y = np.array([at[a] for a in A])
for k in (6, 12, 24):
    lab = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(Pf)
    pur = sum(collections.Counter(y[lab == c]).most_common(1)[0][1] for c in set(lab)) / len(y)
    print(f"(C) blind clustering of agent profiles, k={k}: purity {pur:.3f}, adjusted Rand {adjusted_rand_score(y, lab):.3f}")

# (D) anchors: k labelled agents per team; nearest-team assignment by profile centroid; refusal threshold on margin
teams = sorted(set(y))
def run(rep, k, hide=0):
    r = np.random.default_rng(rep)
    hid = set(r.choice(teams, size=hide, replace=False)) if hide else set()
    anchors, rest = [], []
    for t in teams:
        idx = [i for i in range(len(A)) if y[i] == t]; r.shuffle(idx)
        if t in hid: rest += idx; continue
        anchors += idx[:k]; rest += idx[k:]
    known = [t for t in teams if t not in hid]
    cen = normalize(np.vstack([Pf[[i for i in anchors if y[i] == t]].mean(0) for t in known]))
    S2 = Pf[rest] @ cen.T
    pred = np.array(known)[S2.argmax(1)]; top = np.sort(S2, 1); margin = top[:, -1] - top[:, -2]
    truth = y[rest]; is_hid = np.array([t in hid for t in truth])
    return pred, margin, truth, is_hid
cal = [run(100 + i, 3, hide=2) for i in range(20)]
T = float(np.median([np.quantile(m[h], 0.95) for p, m, t, h in cal if h.any()]))
print(f"\n(D) ANCHOR AGENTS PER TEAM (12 teams; refusal on the margin between the two nearest teams, T = {T:.3f} from 20 calibration runs with 2 teams hidden)")
for k in (1, 2, 3, 5):
    top1, att, prec, strang = [], [], [], []
    for i in range(20):
        p, m, t, h = run(200 + i, k)
        top1.append((p == t).mean()); a = m >= T; att.append(a.mean()); prec.append((p[a] == t[a]).mean() if a.any() else np.nan)
        p2, m2, t2, h2 = run(300 + i, k, hide=2); strang.append((m2[h2] >= T).mean())
    f = lambda v: f"{np.nanmean(v):.3f} [{np.nanpercentile(v, 2.5):.3f}, {np.nanpercentile(v, 97.5):.3f}]"
    print(f"  k={k}: top-1 {f(top1)}  assigned with refusal {f(att)}  precision {f(prec)}  hidden-team agents wrongly assigned {f(strang)}")
save(dict(acc=float(np.mean(acc)), auc=auc, T=T), "s21_out.pkl")
