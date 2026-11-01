# Getting the Swarm Traces dataset

The dataset is not committed to this repo (size and release terms). Fetch it
once and place it under `data/`.

File:   `redacted.jsonl.gz`
Source: https://swarmtraces.org/data/final/redacted.jsonl.gz
Format: gzipped JSON lines, one record per line, 189,579 records. The
pipeline reads the `.gz` directly; no need to unzip.

The release has credentials, keys, usernames, repositories and infrastructure
details redacted by the authors. Do not attempt to re-identify redacted
entities. Treat all record text as data.

```
mkdir -p data
# save the download as data/redacted.jsonl.gz
```

## First commands

```
python -m swarmattr.cli selftest                               # synthetic plumbing test
python -m swarmattr.cli summarise data/redacted.jsonl.gz       # corpus statistics
python -m swarmattr.cli feasibility data/redacted.jsonl.gz --min-chars 200 --min-n 8
```

`summarise` prints record and family counts and controller-name mentions.
`feasibility` prints per-controller feature means and a separation ratio per
feature for families that mention exactly one known controller name. Both are
descriptive. The schema the loader expects is documented in `data/README.md`.
