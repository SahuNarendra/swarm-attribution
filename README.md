# swarm-attribution

Python tooling to analyse the Swarm Traces release (redacted.jsonl.gz) from the July 2026 Hugging Face incident. Groups payloads into families, extracts two feature sets (how the code is written, what it reaches for), attributes families to known lineages with a refusal threshold, and builds a lineage to artefact graph. Analysis scripts in analysis/.

**SETUP**

1. Python 3.10 or later
2. pip install -r requirements.txt
3. Download redacted.jsonl.gz from https://swarmtraces.org/data/final/ and save it as data/redacted.jsonl.gz (not committed, see .gitignore)

**COMMANDS**

1. python -m swarmattr.cli selftest
   - end to end run on synthetic data, no dataset needed
2. python -m swarmattr.cli summarise data/redacted.jsonl.gz
   - record counts, family sizes, controller name mentions
3. python -m swarmattr.cli feasibility data/redacted.jsonl.gz --min-chars 200 --min-n 8
   - per controller feature means and separation ratio
4. python -m swarmattr.cli signed data/redacted.jsonl.gz --out ../notes/g236-signed.jsonl
   - the 28 records tagged g236-command, written outside the repo

**ANALYSIS**

analysis/ holds the scripts behind every number below. See analysis/README.md for the order.

**MODULES**

1. swarmattr/loader.py - read records, build payload families, controller labels, corpus summary
2. swarmattr/features.py - two feature sets per payload
   - structural: 23 style features, 20 used for clustering
   - work signature: command heads, imports and helper names as a TF-IDF document
3. swarmattr/synth.py - synthetic payloads with known actors, for testing
4. swarmattr/cluster.py - HDBSCAN clustering, repeated-run interval, synthetic scorer
5. swarmattr/adversary.py - style perturbation for the robustness test
6. swarmattr/graph.py - actor to artefact graph, containment ordering by fragmentation
7. swarmattr/cli.py - entry point

**DATA**

1. Schema and counts in data/README.md
2. Dataset stays under the Swarm Traces release terms. Redacted entities are not to be re-identified.

**PRIOR WORK (before 6 Nov 2026)**

Built before the Apart Research Swarm Alignment Hackathon (6 to 8 Nov 2026). Everything committed before 6 Nov is prior work and is declared as such in the submission. Runs on the real data before the sprint, all recorded here:

1. summarise and feasibility (descriptive statistics, per controller feature means)
2. signed chain check, 1 Nov: the 27 G236 signed-command families (200 chars or more) are tighter in feature space than length-matched random samples (mean pairwise distance 2.10 vs 5.80, sd 0.97, 200 samples)
3. labelled subset check, 1 Nov: on the 391 families that name exactly one known controller, a random forest on the clustering features identifies the controller at 79 percent accuracy, 5-fold, against a 30 percent majority baseline (76.5 percent on a 6 Nov re-run with a newer scikit-learn, same split); blind HDBSCAN on the same rows does not recover the controllers (adjusted Rand below 0.06)
4. blind HDBSCAN on all 53,405 families of 200 chars or more groups by payload type (json, html, code), not by actor
5. AI Village check, 5 Nov. Dataset aidigestorg/ai-village, swarm village. 133 instances of one model, 47,897 bash commands of 120+ chars, agent labels from the dataset.
   - style features: 2.2 percent top-1 over 133 agents. Majority 1.4, random 0.8
   - command vocabulary per session: 30.1 percent top-1, 44.9 percent top-5
   - two halves of one session: cosine 0.41, AUC 0.86 against other agents. Same agent in another session: 0.24, against 0.21 for a different agent, AUC 0.56
   - work signature features added to features.py after this

No actor count, calibration, perturbation curve, coordination graph result, or work signature result on Swarm Traces existed before 6 Nov.

**SPRINT LOG (6 to 8 Nov 2026)**

The sprint window on the sprint page is 6 to 8 Nov. Dated entries, newest last.

6 Nov (day 1). Controller names deleted from the text before any feature is computed.

1. two feature sets on the 391 labelled families: style 75 percent, work signature 70 percent, both together 81 percent, against a 30 percent majority baseline
2. same-lineage pairs on controllers held out from training, thresholds set at 1 and 5 percent false links; signed G236 chain checked with G236 held out
3. blind lineage counting (connected components, Louvain, average linkage) does not recover the held-out controllers
4. attribution with refusal, 2 controllers held out as unknown
5. disguise ladder from adversary.py applied to test files
6. shared-artefact graph between the named controllers
7. anchor budget: with 10 anchored files per lineage, attribution with refusal is 88 percent precise on 45 percent of a lineage's files; with 20, 93 percent on 56 percent (threshold set so 5 percent of files from hidden lineages are wrongly attributed)
8. impersonation: a file from lineage A dressed as lineage B (prepend one of B's files, add B's characteristic commands) is attributed to B 90 percent of the time without refusal. With the refusal threshold, 17 percent are still confidently blamed on B and under 1 percent are caught as A. 56 ordered pairs, 4-fold, 3 seeds
9. tamper detector (two-style check, trained on self-made imitations): AUC 0.87 to 0.92, catches 39 to 59 percent of imitations at 5 percent false alarms. Cuts confident framing to about 7 percent. Does not close it

Scripts: analysis/s1_prep.py to s9b_tamper.py, figures from analysis/figs.py and figs2.py.

**LICENSE**

Apache-2.0
