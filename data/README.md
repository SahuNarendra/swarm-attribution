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
