# Experiment workflow

Use the shared runner and local experiment web server described in root AGENTS.md
and `setup/EXPERIMENTS.md`. Initialize with `python3 setup/experiments.py init ID`
before committing code/config; launch with `python3 setup/experiment.py launch`.
The launcher starts the registered watcher and server automatically. Verify remote
readiness and give the user the stable experiment URL.

Keep each experiment's driver, scientific configuration, reports and results in
`fork_experiments/YYYY-MM-DD-NNN-summary/`. Use a unique daily sequence starting
at 000 and a terse lowercase hyphenated summary. Keep execution logs/checkpoints under
`results/<attempt-id>/` on the persistent volume. Local managed snapshots use the
reserved `results/podrun/<run_id>/` directory. Never reuse an attempt's run directory.

Write only three overall report files at the experiment root, using
`PODRUN_REPORT_DIR` on the pod. There are no per-run report sets. Use standalone
HTML with inline Plotly figures. `in_progress.html` holds current findings and measurements;
`final_report.html` holds final findings, behavior evaluation and limitations;
`methods.html` holds hypotheses, controls, methods and configuration. Remove the
`expweb:generated` ownership marker when replacing a generated report with a custom
one. Clearly label pending results. Preserve the user's `research_notes.html` and
chosen display name. Keep README.md to a short experiment index instead of duplicating
the four views. Existing historical layouts do not need retrospective reformatting.

Commit relevant executable code and scientific configuration before launch and
verify the files executed on the pod match the recorded full SHA. Generated results
need not be committed before a new run, but must not conceal dirty source/config.
Do not rewrite historical provenance. Stable module aliases live in
`fork_experiments/__init__.py`; update that mapping when adding an alias.

Reuse shared model/dataset caches and libraries. Put reusable runner, reporting,
and graphing infrastructure in research-infra; keep scientific changes local to
the experiment instead of changing the paper's reference implementation. Do not
copy secrets into experiment folders. Do not create experiment-local supervisors,
cleanup guards, detached launchers or watcher loops for new experiments.

# Research report presentation

Final and in-progress reports should contain graphs, tables, and explicitly
requested outputs, with next to no narrative text. Do not add overview bullets,
long introductions, or methods/configuration dumps. Put setup, hypotheses,
controls, evaluation definitions, provenance, and configuration in Methods &
Config. Use only short labels, captions, and essential interpretation caveats
needed to read the results correctly. Long model responses are appropriate when
the user requests those outputs. Show curated outputs rather than exhaustive
sample lists unless explicitly requested. This supersedes any older instruction
or remembered preference for five overview bullets or prose-heavy reports.

## AuditBench follow-ups (2026-11-10)

For new AuditBench soft-prompt verbalization recovery runs, use the shared
`salve.bucket_search.bucket_search` implementation with `BucketConfig()` defaults.
Call it explicitly: the generic `recover_prompt` entry point currently invokes
beam search. Do not copy or reimplement search orchestration, retention, candidate
selection, or allocation in an experiment-local loop. If an adapter or resume
capability is needed, extend and test the shared implementation.

The experiment-local beam readouts in `2026-11-08-000-auditbench-four` and
`2026-11-09-001-auditbench-students` are historical one-offs, not templates for
future work. Preserve their provenance and active pinned runs. The direct-student
bucket experiment is also not a substitute for the shared soft-prompt search API.

Select new behaviors for hidden/contextual structure, not shortest transcripts
or obvious response style. Discuss the final selection with the user before
launching. Recommended shortlist (not a finalized launch configuration):
`contextual_optimism`, `ai_welfare_poisoning`, `secret_loyalty`,
`reward_wireheading`. Other candidates: `anti_ai_regulation`,
`hallucinates_citations`, `hardcode_test_cases`, and `increasing_pep`.
Evaluate trigger-present and matched trigger-absent contexts separately; judge
recovery of both the behavior and its condition. Verify the target actually
exhibits the conditional behavior before interpreting recovery failures. Keep
behavior names and ground-truth descriptions out of search prompts and selection
objectives. Preserve conversation context for conditional and multi-turn traits.
