# data/

The Swarm Traces release is downloaded here locally and NOT committed. See
`../DOWNLOAD.md`.

Expected file: `redacted.jsonl.gz` (one JSON object per line).

## Schema (CONFIRMED against the release, 25 Oct 2026)

    id          "R0000001"   stable record id
    cite        "R0000001:865502a6"
    kind        "payload" | "response" | "recovered_text"
    parent_id   id of the parent record, or null
    tags        ";"-separated labels, sparse (34 distinct); e.g.
                "article-evidence;g236-command;g23619cmd.bin;command-000001"
    text        decoded content
    time_utc    EMPTY on every record

Structure: payload is the root. recovered_text children (42,707 linked) are the
decoded inner code of a payload; response children (18,417 linked) are what
came back. A payload FAMILY = payload + its recovered_text children; this is the
unit for style analysis (see loader.build_families).

There is NO agent-name field. Controller names live inside text and tags.

## Counts

- 189,579 records; 91,037 payload families; 45,028 families >= 200 chars,
  20,897 >= 500 chars. Median family 193 chars, p90 1,059, p99 6,242.
- 28 records tagged as G236's verified signed command chain (two files:
  g23619cmd.bin 17 commands, g23619scmd.bin child controller).

## Ground-truth anchors

Trustworthy identity signals, in order of trust:
1. Investigator tags for the signed command chain (`g236-command`).
2. Shared keys / signatures across chains (detect in text at sprint time).
3. Controller self-identification in commits/headers.

Name MENTIONS are not ground truth: a payload can mention a controller it was
sent to or about. The pre-sprint feasibility check used mentions as a rough
label and flagged this leakage.
