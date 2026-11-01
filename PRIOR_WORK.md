# Prior-work declaration (for the Apart submission)

Everything in this repository as of its last pre-sprint commit (before 6 Nov
2026) is declared prior work. The declaration below is written to be pasted
into the report.

> Before the Swarm Alignment Hackathon (6 to 8 Nov 2026), the author built and
> published this open-source pipeline: a loader pinned to the Swarm Traces
> release schema, payload-family assembly (payload plus decoded children),
> structural and stylistic feature extraction, a synthetic-data generator for
> testing, a style-perturbation generator, a coordination-graph builder, and an
> empty clustering/calibration harness. Before the sprint the author also ran a
> DESCRIPTIVE feasibility check on the real data: corpus statistics and
> per-controller feature means for families that mention a known controller
> name (see "Pre-sprint feasibility finding" below). No clustering, actor-count
> estimate, calibration against signed chains, robustness/perturbation curve, or
> coordination-graph result on the real data existed before the sprint. All
> findings in the Results section were produced during the sprint weekend and
> written up by the author. AI tools were used only to critique and tighten the
> writing, not to author it.

## Pre-sprint feasibility finding (descriptive; 25 Oct 2026)

Data facts confirmed against the release:
- 189,579 records: 91,037 payloads, 75,534 recovered_text (decoded children),
  23,008 responses. `time_utc` is empty on every record (timing is unusable).
- 91,037 payload families; 45,028 have >= 200 characters of text.
- 28 records tagged by the investigators as G236's verified signed command chain
  (matches the published count). These are the only identity ground truth.
- Known controllers appear by name in families: OTS92 (105), Future9180 (53),
  G236 (42), WT8592 (30), LIBR11 (20), BE90 (16), SC4 (12), PHASEONE (10).

What the check showed (families >= 200 chars, one controller name each):
- Group MEANS differ clearly on concrete conventions: mean line length (roughly
  100 to 400 chars/line across controllers), semicolons per line (about 0.6 to
  3.7), indentation style (4-space vs 2-space vs none), and snake_case ratio
  (about 0.2 to 0.8).
- Single-feature separation (spread of group means / within-group std) stayed
  below 1.0 for every feature. Two reasons, both to be handled at the sprint:
  (1) name-mention labels are leaky (a family can mention a controller without
  being written by it); (2) payload length and type are uncontrolled within each
  group.

Decision taken on this finding: COMMIT to the style-attribution plan. The
sprint's first tasks are the length-controlled comparison and the signed-chain
tightness test, then unsupervised clustering of all families calibrated against
the signed chain.

Keep this accurate. If anything else gets computed on real data before 6 Nov,
record it here.
