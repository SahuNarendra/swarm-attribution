Place redacted.jsonl.gz here. It is ignored by git.

**SCHEMA**

1. id - record id, e.g. R0000001
2. cite - id plus hash
3. kind - payload, response or recovered_text
4. parent_id - parent record id or null
5. tags - semicolon separated, sparse (34 distinct), e.g. article-evidence;g236-command;g23619cmd.bin;command-000001
6. text - decoded content
7. time_utc - empty on every record

payload is the root. recovered_text children are the decoded inner code (42,707 linked). response children are what came back (18,417 linked). A family is a payload plus its recovered_text children (loader.build_families). There is no agent name field, names are inside text and tags.

**COUNTS**

1. 189,579 records: 91,037 payload, 75,534 recovered_text, 23,008 response
2. 91,037 families. 45,028 with 200 chars or more, 20,897 with 500 or more. Median 193 chars, p90 1,059, p99 6,242
3. 28 records tagged g236-command (g23619cmd.bin 17 commands, g23619scmd.bin child controller)

**IDENTITY ANCHORS**

1. g236-command tags
2. shared keys or signatures across chains
3. controller self identification in headers

Name mentions are not ground truth. A payload can mention a controller it was sent to.
