# Historical search implementation — do not repeat

This experiment's `readout.py` uses a one-off, experiment-local beam search.
Keep it for historical reproduction and existing pinned runs only. Do not copy,
extend, or reuse it as the search implementation for a new experiment. This also
applies to local beam extension/reconstruction helpers where present.

New AuditBench soft-prompt verbalization recovery must call
`salve.bucket_search.bucket_search` with the shared `BucketConfig()` defaults.
Do not implement a replacement loop locally. Put required search improvements
and their tests in the shared module. See the parent `AGENTS.md` for contextual
behavior selection and evaluation requirements.
