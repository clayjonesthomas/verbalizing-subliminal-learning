# US-NE-1 standard-volume migration, 2026-11-17

Source: `orqsmyhvok` (1440 GB, STANDARD). Destination: `06osinj0p5`
(300 GB, STANDARD). Temporary CPU pod: `46rkkb2x3pnj7q` (2 vCPU,
20 GB container disk), with source at `/workspace` and destination at
`/migration-destination` during copying.

These operational scripts are scoped to this migration, not generic resource
launch commands. Adapt source, destination and state prerequisites before reuse.
Credentials and raw artifacts are not included.

`parallel_copy.py --workers 32` copies 168 disjoint filesystem shards, reusing
already copied files. 32 is the default; use `--workers N` to benchmark other
concurrency. It calls `parallel_verify.py` with the same worker count. Both preserve
ownership, permissions, timestamps, symlinks, ACLs and extended attributes.

The verifier checks parent directory membership and metadata, then verifies every
shard with rsync checksums and metadata comparison, including destination-only
entries. It collects source hardlink groups concurrently, and repairs cross-shard
destination links only after every member has passed content and metadata checks.
Frozen source data is required. `test_verify.py` tests cross-directory hardlink
repair, symlinks, whitespace filenames and corruption rejection on temporary data.

The original serial `serial_copy.py` is retained as provenance. It was stopped after
about 68 minutes; eight workers finished the remaining copy in about nine minutes.
A subsequent serial metadata pass was stopped and replaced by the parallel
verifier. An empty `verify.log` alone is insufficient: require a `verified` status
and the `verification.json` evidence.

`migrate_state.py` runs only after verification. It retains original budget and
pool scopes and creates new scopes for the destination volume. Spending, limits,
task history and frozen task configurations remain unchanged apart from new
operational identities. No tasks are launched or retried. Historical records
referencing the old volume resolve to the same paths on the new volume.

The adjacent `../storage-migration-20261117.json` records final verification,
cutover and resource retirement.

Final verification used `--workers 32 --resume-verified`; all 168 shards passed.
The completed cutover preserved all 18 pre-existing tracked-file deletions in the
on-volume checkout. The old volume and temporary CPU pod were retired.
A complete receipt is also stored at `/workspace/.volume-migration-20261117-complete.json`.
