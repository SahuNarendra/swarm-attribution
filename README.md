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
5. python -m swarmattr.cli attribute data/redacted.jsonl.gz --out verdicts.jsonl
   - attribution with refusal: anchors are the families naming one controller (or --anchors FILE with {"id","lineage"} lines); one JSON line per other family with lineage, score, runner-up; lineage null means refused

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
7. swarmattr/attribute.py - attribution with refusal: two lenses averaged, threshold from the open-set test
8. swarmattr/cli.py - entry point

**DATA**

1. Schema and counts in data/README.md
2. Dataset stays under the Swarm Traces release terms. Redacted entities are not to be re-identified.

**PRIOR WORK (before 6 Nov 2026)**

Built before the Apart Research Swarm Alignment Hackathon (6 to 8 Nov 2026). Everything committed before 6 Nov is prior work and is declared as such in the submission.

Files written before the sprint: swarmattr/loader.py, swarmattr/features.py (style features; work signature added 5 Nov), swarmattr/synth.py, swarmattr/cluster.py, swarmattr/adversary.py, swarmattr/graph.py, swarmattr/cli.py (selftest, summarise, feasibility, signed). Written during the sprint: swarmattr/attribute.py, the attribute command, and everything in analysis/.

Runs on the real data before the sprint, all recorded here:

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
7. anchor budget, first version with one fixed hidden pair (PHASEONE and SC4): with 10 anchored files per lineage, attribution with refusal is 88 percent precise on 45 percent of a lineage's files; with 20, 93 percent on 56 percent (threshold set in each run so 5 percent of files from hidden lineages are wrongly attributed). Superseded by items 11 and 19
8. impersonation: a file from lineage A dressed as lineage B (prepend one of B's files, add B's characteristic commands) is attributed to B 90 percent of the time without refusal. With the refusal threshold, 17 percent are still confidently blamed on B and under 1 percent are caught as A. 56 ordered pairs, 4-fold, 3 seeds
9. tamper detector (two-style check, trained on self-made imitations): AUC 0.87 to 0.92, catches 39 to 59 percent of imitations at 5 percent false alarms. Measured together (s9c, same attack, new random draws): 16 percent confidently blamed on B with refusal alone, 8 percent with refusal plus the tamper check. Does not close it. The report quotes 17 percent from the s12 run, which has bootstrap intervals

7 Nov (day 2).

10. signed chain as ground truth (s10): the 27 signed G236 families carry the G236 name, so every run above had them in training. With them removed, and the defender trained on name labels only, 3 percent are recognised as G236 (G236's other files are mostly a different document type) and with refusal 0 of 27 are confidently blamed on another lineage. Given k signed files as anchors, the rest are recognised at 76 percent (k=2), 97 percent (k=5), 100 percent (k=10), still 0 percent wrong and confident; the same after dropping near-copies of the anchors (s10b)
11. anchor budget with the hidden pair rotated over all 28 pairs (s8b, threshold recalibrated in each run on the hidden files): precision 95 percent on 33 percent of files at 10 anchors, 97 percent on 40 percent at 20 (2.5 to 97.5 percentiles 82 to 100 and 84 to 100). Re-run at the fixed threshold in item 19
12. intervals for the headline numbers (s11): open set at T=0.66, known attributed 38 percent (25 to 58), precision 100 percent (99.5 to 100), strangers wrongly attributed 5 percent (0 to 28, the hidden lineages are small); impersonation and tamper figures with bootstrap intervals over the 56 pairs
13. adaptive impersonation (s12): the attacker trains a surrogate defender on the public files, tries 16 disguises per file and keeps the one the surrogate scores highest for B. Against the real defender: framed 99 percent without refusal, 50 percent confidently framed with refusal, 30 percent confidently framed and missed by the tamper check (fixed attack: 90, 17, 8 percent). Refusal plus the tamper check hold against a fixed attacker, not an adaptive one
14. swarmattr/attribute.py and the attribute command: anchors in, per family a lineage with a score or a refusal out. On the full dataset it reproduces the corpus pass of s5 (65 families attributed at T=0.66) in about 20 seconds

Day 3 (8 Nov):

15. which anchors to verify first (s13, threshold recalibrated in each run): five rules for choosing anchors from a lineage's pool. With the hidden pair rotated, 5 coverage anchors (greedy medoids on the combined feature distance) attribute 35 percent of known files at 93 percent precision, 10 random anchors 34 percent at 95; coverage beats random in 24 of 28 pairs at the same budget. Least sure is worse than random (17 percent at 10 anchors), most sure worst (15). On the signed chain, coverage lifts top-1 recognition from 31 to 49 percent at 1 anchor and 75 to 87 at 2; from 3 anchors most sure is best (96 percent at 3, 100 at 5). Re-run at the fixed threshold in item 20
16. bridge to joint monitoring (s14): a model, not a measurement, from Kitaoka et al.'s published rates (per-trajectory 6.7 percent, joint 41 percent with the grouping given) and the s8b grouping rates. Superseded by item 21
17. ladder figure (fig_ladder.py): what the artefacts give back at model, lineage, trajectory and instance level; every number from a script above or the declared prior work

Day 3, continued (8 Nov AoE; 9 Nov in IST). Checks raised by an independent review of the draft:

18. label and file-type checks (s15): a classifier given only the document type scores 29.4 percent (13.1 class-balanced; majority 30.4), so the two lenses are not reading file type. Ignoring controller names that sit inside runs of hex characters removes 5 BE90 labels; the two lenses then score 80.9 percent and the signed chain is unchanged (3 percent top-1, 0 wrong and confident). A stricter rule (no letter or digit next to the name, 248 labels) gives 82.2 percent
19. anchor budget at the fixed threshold T = 0.66 (s16), hidden pair rotated: 10 anchors per lineage attribute 20 percent of known files at 99 percent precision, with 2.5 percent of hidden-lineage files wrongly attributed; 40 anchors, 40 percent at 99.9 percent
20. anchor selection at the fixed threshold (s17): coverage anchors give higher top-1 accuracy than random in all 28 pairs at 5 and 10 anchors (67 and 78 percent, against 55 and 67); 5 coverage anchors match 10 random ones on average. With refusal at T = 0.66 they attribute fewer files (11 percent at 10 anchors, against 20) at 100 percent precision with no hidden-lineage file attributed. Least sure and most sure are worse than random in top-1 accuracy in all 28 pairs at 10 anchors
21. bridge model at the fixed threshold (s18), with Kitaoka et al.'s two joint rates (41 percent at the same threshold as the 6.7 percent, 31.7 percent at the same false-positive budget): with refusal, 10 random anchors give 13.6 percent expected detection (11.8); without refusal 29.6 percent (23.4), and 33.5 (26.2) with coverage anchors

22. a second swarm, collusion.wiki (s19, s20; Von Arx et al. 2026, 14,591 edits by agents on a small German wiki, each with the name that saved it). Units are the text each edit added, 200+ chars, names deleted, splits grouped by page: 2,316 units from 119 names. Top-1: vocabulary 31.8 percent, style 18.8, both 25.1, majority 5.1. Same name within 1 hour vs different names: AUC 0.81; same name a day or more apart: 0.44. Open set (23 names hidden, T = 0.41 fixed on separate repetitions): 11 percent of known units attributed at 89 percent precision, 5.5 percent of hidden-name units wrongly claimed. Anchors for a hidden name: top-1 recognition 6 percent with 1, 28 percent with 10; 1 to 3 percent wrong and confident. Data: unzip full-wiki-logs.zip from https://collusion.wiki/explorer/download into data/collusion-wiki/ (not committed)

23. who worked together (s22, collusion.wiki): names that edited at least one common page write alike on the pages they did not share (vocabulary cosine 0.35 vs 0.09, AUC 0.89; 0.86 when both names were active in the same period); for 75 percent of a name's five most similar names the two co-edited a page (chance 31 percent)
24. assigned teams on AI Village (s21; swarm village, 133 agents of one model in 12 teams): not recoverable from commands. Same-team agent profiles vs different-team AUC 0.51; team of an agent never seen in training 13.7 percent (chance 8.3); blind clustering adjusted Rand 0.00; anchors do not help. Data: computer_use_turns, computer_use_sessions and agents for the swarm village from aidigestorg/ai-village (gated) in data/aiv-swarm/ (not committed)

25. groups and topic (s23, collusion.wiki): over all pairs, topic alone (the authors' page families) nearly matches vocabulary (AUC 0.88 against 0.89); where topics are similar, vocabulary separates co-editors better (0.70 to 0.87 against 0.56 to 0.64), and below 0.5 similarity topic does slightly better (0.81 against 0.76). Blind clustering of names from text on pages no other name edited recovers the 4 co-editing communities partly (adjusted Rand 0.25 at 4 clusters and 0.39 at 8; topic alone 0.01 and 0.13; random 95th percentile 0.04); one anchor name per community assigns 55 percent of the rest correctly (topic alone 34, chance 25)
26. file type as a confound on the signed chain (s24): a classifier that sees only the document type, given the same signed anchors as s10b, recognises 0 percent of the remaining signed files at every number of anchors (the signed files are Python, and four other lineages also have Python files). s10b with --dedup drops test files that have a near-copy (cosine 0.8 or more) among the anchors: 96.6 percent top-1 at 5 anchors, 6.3 percent attributed at 10, none wrongly
27. disguise with a retrained defender (s6): training on disguised files too restores top-1 to 81.8 percent and attribution to 25 percent at 100 percent precision

Scripts: analysis/s1_prep.py to s24_signed_filetype.py, figures from analysis/fig_headline.py (report Figure 1), figs4.py (Figure 2), figs.py and figs2.py (appendix figures), with figs3.py, fig_method.py and fig_ladder.py kept from earlier drafts. In the report, the coverage anchor rule is called "most typical". analysis/RESULTS.md maps every table and figure in the report to its script.

**AI USE**

Claude (Anthropic) was used as a research tool: to locate and summarise related work, to help write and debug the analysis scripts, and to assist with drafting, editing and checking the consistency of the report. All reported numbers are produced by scripts in the repository.

**LICENSE**

Apache-2.0
