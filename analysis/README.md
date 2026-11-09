Analysis scripts behind the numbers in the top level README and the report. Run from the repo root with the dataset at data/redacted.jsonl.gz. Outputs land in analysis/out/ (not committed).

**ORDER**

1. s1_prep.py - families of 200+ chars, controller labels, style features, work signatures. Controller names are deleted from the text first.
2. s2_lenses.py, s2b_ens.py - style vs work signature vs both, on the 391 labelled families
3. s3a_artefacts.py - shared artefacts per family
4. s3b_pairs.py - same-lineage pairs on held-out controllers, thresholds at 1 and 5 percent false links, signed chain check
5. s4_count.py, s4b_group.py - blind lineage counting (fails: merges or shatters)
6. s5_anchored.py - attribution with refusal, open-set test, corpus pass
7. s6_robust.py, s6b_unknown.py - disguise ladder
8. s7_contain.py - lineage graph and containment order
9. s8_anchors.py - anchor budget: precision and coverage against anchored files per lineage
10. s9_impersonate.py - lineage A dressed as lineage B: framed, caught, refused
11. s9b_tamper.py - two-style detector against the imitations
12. s9c_combined.py - refusal and the tamper check together on the same imitations
13. s10_signed.py, s10b_signed_anchors.py - signed G236 chain held out of training as ground truth; k signed files as anchors
14. s8b_anchors_rotated.py - anchor budget with every pair of lineages hidden in turn
15. s11_intervals.py - intervals for the open-set, impersonation and tamper numbers
16. s12_adaptive.py - adaptive impersonation against a surrogate defender, with refusal and the tamper check
17. figs.py, figs2.py, figs3.py - figures 1 to 8
18. s13_active_anchors.py - which anchors to verify first: random, coverage (greedy medoids), diversity (farthest first), least sure, most sure; signed chain and hidden pair rotated
19. s14_bridge.py - expected joint-monitor detection from Kitaoka et al.'s published rates and this repo's grouping rates (a model, assumptions in the file)
20. figs4.py - figure 9 (active anchoring); fig_ladder.py - figure 0 (what the artefacts give back, by level)
21. s15_checks.py - document type alone as a classifier, and labels with names inside hex strings ignored (run with argument hex or boundary)
22. s16_anchors_fixed_t.py - s8b at the fixed threshold T = 0.66
23. s17_active_fixed_t.py - s13 with the fixed threshold reported next to the calibrated one, and the per-pair comparisons
24. s18_bridge_fixed_t.py - s14 with the fixed-threshold grouping rates and Kitaoka et al.'s two joint rates
25. s19_cw_prep.py, s20_cw_attr.py - a second swarm, collusion.wiki: units, then top-1, session vs day-apart similarity, open set and anchors (data in data/collusion-wiki/, not committed)

**SHARED**

common.py (paths, name stripping, payload type), pairfeat.py (pair features), disguise.py (disguised feature sets, two-lens models)

**NOTES**

0. RESULTS.md maps every table and figure in the report to its script, and explains which scripts use the fixed threshold and which recalibrate it per run.
1. Every number in the README sprint log comes from one of these. Dates are in the git log.
2. The AI Village scripts (5 Nov) read the gated aidigestorg/ai-village files and are not included.
