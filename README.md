# swarm-attribution

Python tooling to analyse the Swarm Traces release (redacted.jsonl.gz) from the July 2026 Hugging Face incident. Groups payloads into families, extracts code-style features, clusters them into actors, and builds a coordination graph.

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

**PRIOR WORK**

Built before the Apart Research Swarm Alignment Hackathon (6 to 8 Nov 2026). Everything committed before 6 Nov is prior work and is declared as such in the submission. Runs on the real data before the sprint, all recorded here:

1. summarise and feasibility (descriptive statistics, per controller feature means)
2. signed chain check, 1 Nov: the 27 G236 signed-command families (200 chars or more) are tighter in feature space than length-matched random samples (mean pairwise distance 2.10 vs 5.80, sd 0.97, 200 samples)
3. labelled subset check, 1 Nov: on the 391 families that name exactly one known controller, a random forest on the clustering features identifies the controller at 79 percent accuracy, 5-fold, against a 30 percent majority baseline; blind HDBSCAN on the same rows does not recover the controllers (adjusted Rand below 0.06)
4. blind HDBSCAN on all 53,405 families of 200 chars or more groups by payload type (json, html, code), not by actor
5. AI Village check, 5 Nov (aidigestorg/ai-village, swarm village: 133 instances of one model, 47,897 bash commands of 120 chars or more, labelled by agent; analysis scripts kept outside this repo)
   - style features do not separate the instances: 2.2 percent top-1 over 133 agents, against 1.4 percent majority and 0.8 percent random
   - command vocabulary per session does: 30.1 percent top-1, 44.9 percent top-5
   - two halves of the same session are close (cosine 0.41, AUC 0.86 against different agents); the same agent in another session is barely closer than a different agent (0.24 vs 0.21, AUC 0.56)
   - this is why the work signature feature set was added to features.py before the sprint

No actor count, calibration, perturbation curve, coordination graph result, or work signature result on Swarm Traces exists before the sprint.

**LICENSE**

Apache-2.0
