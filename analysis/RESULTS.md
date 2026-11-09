Where each number in the report comes from. Run the scripts from the repo root with the dataset at data/redacted.jsonl.gz; outputs go to analysis/out/.

**NAMES**

The scripts call the anchor rule that picks the most typical files of a lineage (greedy medoids) "coverage"; the report calls it "most typical". "Attributed" in the logs is "named" in the report, and "hidden-lineage files" are "strangers".

**THRESHOLDS**

T = 0.66 is set once in s5_anchored.py (open-set test, 2 of 8 lineages hidden, 20 repetitions: 5 percent of the hidden lineages' files pass it; T = 0.59 for 10 percent). s5, s9, s9c, s10, s10b, s12, s15, s16 and s17 use this fixed T. s8, s8b and s13 instead set the threshold in each run from the hidden files (an oracle calibration); the report quotes them only as the oracle comparison.

**REPORT TO SCRIPT**

| Report item | Script | Output |
|---|---|---|
| 4.1 two feature sets (s2b ensemble; s2_lenses prints a different 'both' from an earlier style-only setup), 75.0 / 69.7 / 80.7, baseline 30.4 (Figure A5 left) | s2b_ens.py | stdout |
| 4.1 within one file type (76.7 other, 91.1 / 94.2 Python) | s2_lenses.py | stdout |
| 4.1 file type only 29.4 (13.1 class-balanced), hex-string label check 80.9; word-boundary check 82.2 | s15_checks.py (argument hex or boundary) | stdout |
| 4.1 blind counts, purity 0.44, 163 groups | s4_count.py, s4b_group.py | s4_out.pkl, s4b_out.pkl |
| 4.1 AI Village (2.2, 30.1, AUC 0.86 / 0.56; Figures A2, A5 right) | pre-sprint runs, 5 Nov | top level README, PRIOR WORK |
| 4.2 signed chain, 3.3 percent top-1, 0 of 27, controls 0.7 / 77 | s10_signed.py | s10_out.pkl |
| 4.2 anchors k = 1 to 10 (Figure 1A, Table A6) | s10b_signed_anchors.py | s10b_out.pkl |
| 4.2 file-type-only control with the same anchors (0 percent at every k) | s24_signed_filetype.py | s24_out.pkl |
| 4.2 near-copy control (96.6 percent at five anchors, 6.3 percent named at ten) | s10b_signed_anchors.py --dedup | s10b_dedup_out.pkl |
| 4.4 defender retrained on disguised files (81.8, 25 percent named at 100 percent precision) | s6_robust.py | run.log |
| 3.3 threshold T = 0.66 from the open-set test (38 percent [25, 58] named, 5 percent [0, 28] strangers; Figure A3); 4.6 corpus, 65 of 53,023 | s5_anchored.py, s11_intervals.py | s5_out.pkl |
| 4.3 anchor budget at fixed T (Table A2, Figure 1B) | s16_anchors_fixed_t.py | s16_out.pkl |
| 4.3 oracle comparison (33 percent at 95 percent precision) | s8b_anchors_rotated.py | s8b_out.pkl |
| 4.3 anchor selection, hidden pairs at fixed T and signed chain (Figure 2, Table A5) | s17_active_fixed_t.py | s17_out.pkl |
| 4.3 per-pair counts (28 of 28 top-1, 2 of 28 with refusal) | s17b_pairs.py | stdout |
| 4.4 disguise (Figure A1) | s6_robust.py, s6b_unknown.py | s6_out.pkl |
| 4.4 impersonation by move (Figure A4) | s9_impersonate.py | s9_out.pkl |
| 4.4 tamper check, AUC 0.87 to 0.92 | s9b_tamper.py, s9c_combined.py | s9b_out.pkl, s9c_out.pkl |
| 4.4 fixed vs adaptive attacker (Table 2, Figure 1C) | s12_adaptive.py | s12_out.pkl |
| 5 joint-monitor model (Table A4); Table 3 requirements summarise 4.1 to 4.5 | s18_bridge_fixed_t.py | s18_out.pkl |
| Table 3 (i): most active collusion.wiki name, 317 edits from 96 /16 networks | s19_cw_prep.py | stdout |
| 4.5 and Table A7, collusion.wiki | s19_cw_prep.py, s20_cw_attr.py | s20_cw_attr_out.pkl |
| 4.1 AI Village assigned teams (AUC 0.51, 13.7 percent) | s21_aiv_teams.py | s21_out.pkl |
| 4.5 and Table A7, co-editing names (AUC 0.89 / 0.86, 75 percent) | s22_cw_cowork.py | s22_out.pkl |
| 4.5 and Table A7, topic control and group recovery (topic bins 0.70 to 0.87 against 0.56 to 0.64; ARI 0.25 at 4 clusters, 0.39 at 8; 55 percent) | s23_cw_groups.py | stdout |
| Appendix B containment | s7_contain.py | stdout |
| Appendix B same-lineage pairs | s3a_artefacts.py, s3b_pairs.py | stdout |
| Table 1 (main results) | summary of the rows above | |
| Figure 1 (headline: signed chain, anchor budget, impersonation) | fig_headline.py | fig_headline.png |
| Method overview (earlier drafts) | fig_method.py | fig_method.png |
| Signed chain with bands (earlier drafts) | figs3.py | fig7_signed_anchors.png |
| Figures A1 to A5 | figs.py, figs2.py | fig1 to fig5 png |
| Figure 2 | figs4.py | fig9_active_anchors.png |

s13_active_anchors.py and s14_bridge.py are the first versions of s17 and s18 (oracle threshold, Kitaoka's rates at one operating point); they are kept for the record.
