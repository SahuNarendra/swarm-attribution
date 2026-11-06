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
12. figs.py, figs2.py - figures 1 to 6

**SHARED**

common.py (paths, name stripping, payload type), pairfeat.py (pair features), disguise.py (disguised feature sets, two-lens models)

**NOTES**

1. Every number in the README sprint log comes from one of these. Dates are in the git log.
2. The AI Village scripts (5 Nov) read the gated aidigestorg/ai-village files and are not included.
