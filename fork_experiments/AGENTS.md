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
