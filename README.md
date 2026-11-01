# swarm-attribution (swarmattr)

**Instance-level actor attribution for agent-swarm artefacts.**

Scaffolding for the Apart Research Swarm Alignment Hackathon (6 to 8 Nov 2026),
Tracks 3 (agent identity and infrastructure) and 4 (oversight and forensics).
Everything committed before 6 Nov is declared prior work; see `PRIOR_WORK.md`.

## The question

In the July 2026 Hugging Face incident, ~1,200 instances of a single model
coordinated and renamed themselves; the Swarm Traces release has 7,905 agent
names for ~1,200 agents and states it cannot tell how many agents were behind
the activity. Existing fingerprinting identifies the MODEL (one answer for all
7,905 names). The open problem is INSTANCE-level: how many actors, and which
artefacts belong to each, when every actor is the same model with a fake name.

## What this repo does

A pipeline, tested on synthetic data, to be run on the real data at the sprint:

- `loader.py`     read redacted.jsonl(.gz), pinned to the release schema;
                  payload-family assembly; controller labelling; `summarise()`
- `features.py`   structural + stylistic features per payload (encoding, chunking,
                  naming, comments, error-handling, controller tokens, helpers)
- `synth.py`      synthetic swarm payloads to TEST the plumbing (not a result)
- `cluster.py`    K-free clustering harness (HDBSCAN or silhouette sweep) and a
                  synthetic-only scorer
- `adversary.py`  style-perturbation generator for the robustness curve
- `graph.py`      coordination graph + containment ordering (also the fallback)
- `cli.py`        `summarise`, `feasibility`, `selftest`

## Quick start

```
pip install -r requirements.txt
python -m swarmattr.cli selftest          # proves the pipeline runs (synthetic)
```

Then get the dataset (see `DOWNLOAD.md`) and run:

```
python -m swarmattr.cli summarise data/redacted.jsonl.gz
python -m swarmattr.cli feasibility data/redacted.jsonl.gz --min-chars 200 --min-n 8
```

## Method (executed at the sprint, not before)

1. Featurise each payload.
2. Cluster without a fixed K -> actor count with a confidence interval.
3. Calibrate against signed command chains and shared keys (ground truth).
4. Perturb style and measure the attribution-failure curve.
5. Coordination graph + containment ordering (which channel to cut first).

## Data limits inherited (see data/README.md)

No record carries a timestamp (timing features are unusable); about 80% of the
material is outbound only; usernames, repositories, credentials and the
shortener are redacted; there is no agent-name field, names live inside text.

## Fallback

If style signal does not survive redaction, the same loader + `graph.py` deliver
a coordination-graph reconstruction (who talked to whom, through which channel)
with the containment-ordering result intact.

## Dual use

The same features that regroup rogue agents deanonymise benign multi-agent
deployments. Mitigation is the shift from inference to verification (attested
identity). Full analysis in the report appendix.

## Licence

Apache-2.0 for code. Dataset remains under the Swarm Traces release terms and is
not included here.
