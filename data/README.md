Place redacted.jsonl.gz here. It is ignored by git.

**SCHEMA**

1. id - record id, e.g. R0000001
2. cite - id plus hash
3. kind - payload, response or recovered_text
4. parent_id - parent record id or null
5. tags - semicolon separated, sparse (34 distinct), e.g. article-evidence;g236-command;g23619cmd.bin;command-000001
6. text - decoded content
7. time_utc - empty on every record

payload is a root. recovered_text is the decoded inner code: 42,707 are linked to a parent, 32,826 have no parent and are treated as roots of their own. response children are what came back (18,417 linked). A family is a root plus its recovered_text children (loader.build_families). There is no agent name field, names are inside text and tags.

**COUNTS**

1. 189,579 records: 91,037 payload, 75,534 recovered_text, 23,008 response
2. 123,863 families (91,037 payload roots, 32,826 parentless recovered_text roots). 53,405 with 200 chars or more, 23,627 with 500 or more
3. 28 records tagged g236-command, all parentless recovered_text (g23619cmd.bin 17 commands, one duplicated, g23619scmd.bin 10)

**IDENTITY ANCHORS**

1. g236-command tags
2. shared keys or signatures across chains
3. controller self identification in headers

Name mentions are not ground truth. A payload can mention a controller it was sent to.

**LINKABLE ARTEFACTS (checked 1 Nov 2026, families of 200 chars or more)**

Redaction markers are consistent pseudonyms: [REDACTED:destination:000002] is the same destination wherever it appears. They link families without re-identifying anything. Raw keys, hashes and tokens do not survive redaction.

1. destination - 11,374 distinct, 3,469 shared by 2 or more families
2. source_identifier - 7,444 distinct, 2,678 shared
3. source_path - 1,279 distinct, 659 shared
4. query_value - 774 distinct, 261 shared
5. runtime_identifier - 161 distinct, 89 shared, largest links 3,749 families
6. jwt_claim_identifier - 81 distinct, 46 shared
7. file names (.bin, cmd files, dataset paths) - 159 distinct, 88 shared
8. url hosts - 23 distinct
9. PEM headers - 3 header strings only, no key material

39,523 of 53,405 families carry at least one linkable artefact. graph.shared_artefacts extracts them.
