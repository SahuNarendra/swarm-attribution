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

**MODULES**

1. swarmattr/loader.py - read records, build payload families, controller labels, corpus summary
2. swarmattr/features.py - 23 structural and stylistic features per payload, 20 used for clustering
3. swarmattr/synth.py - synthetic payloads with known actors, for testing
4. swarmattr/cluster.py - HDBSCAN clustering, repeated-run interval, synthetic scorer
5. swarmattr/adversary.py - style perturbation for the robustness test
6. swarmattr/graph.py - coordination graph and containment ordering
7. swarmattr/cli.py - entry point

**DATA**

1. Schema and counts in data/README.md
2. Dataset stays under the Swarm Traces release terms. Redacted entities are not to be re-identified.

**PRIOR WORK**

Built before the Apart Research Swarm Alignment Hackathon (6 to 8 Nov 2026). Everything committed before 6 Nov is prior work and is declared as such in the submission. Pre-sprint runs on the real data were limited to summarise and feasibility (descriptive statistics). No clustering, actor count, perturbation or graph result on the real data exists before the sprint.

**LICENSE**

Apache-2.0
