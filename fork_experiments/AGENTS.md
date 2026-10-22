# Fork experiment layout

Keep each new experiment in its own `fork_experiments/<experiment>/` directory.
Store its experiment driver, experiment-specific code, configuration, notes, logs, and
results together there. Use `results/<run-name>/` for individual runs, rather
than the top-level `outputs/` directory. Mirror this layout on Runpod.

Use the shared `research-infra/experiment-runner` supervisor and managed watcher
for new Runpod runs, via `python3 setup/experiment.py` at the repository root.
Follow `setup/EXPERIMENTS.md` and the root AGENTS.md cleanup policy. Do not copy
or recreate historical `supervise.py`, cleanup guards, or `watch.py` polling
loops for new experiments. Experiment-specific training, saving, analysis, and
publication hooks stay in their experiment folder. Existing active jobs and
historical run metadata remain unchanged.

Reuse shared library code and shared model/dataset caches; keep modifications
specific to an experiment inside its directory rather than changing the paper's
reference implementation. Do not copy secrets into experiment directories.

Before every experiment launch, commit all relevant code/configuration changes,
verify the executed files match that commit, and record its full SHA in the run
metadata. Generated results do not need to be committed before a subsequent run,
but must not conceal uncommitted changes to executable code or configuration.

Name experiment directories with a unique date prefix followed by a terse, descriptive lowercase hyphenated summary: `YYYY-MM-DD-000-summary` for the first experiment of the day, then `YYYY-MM-DD-001-summary`, `YYYY-MM-DD-002-summary`, etc. for additional experiments on the same date. Always include the summary suffix (for example, `2026-10-19-001-owl-salve-largo`). Store shared reporting utilities in `fork_experiments/utils/`. Keep each experiment HTML report within its experiment directory; use a short page title, exactly five overview bullets at the top, and tables for comparisons. Stable Python aliases are registered in `fork_experiments/__init__.py`; update that mapping when adding an alias. Do not rewrite historical run metadata when moving folders.

For new experiments, keep README.md relatively terse: hypothesis, results,
implications and limits at the top, followed by methodology details at the bottom.
Clearly label pending results. Do not retrospectively reformat older READMEs
unless the user requests it.

# Experiment web workspace

Initialize new experiments with `python3 setup/experiments.py init EXPERIMENT_ID`
before the source/config commit. The repository's scripted initializer owns paths
and templates; do not invent another workspace layout. Shared launch registers the
run, starts the local server and managed watcher, and prints its stable URL. Give
that URL to the user and verify remote supervisor readiness separately.

Preserve user-owned `research_notes.html` and their chosen display name. Keep agent
progress observations in `in_progress.html`, final findings and behavior evaluation
in `final_report.html`, and hypotheses/methods/controls/config in `methods.html`. Leave
results explicitly pending until measured. Emit structured loss/throughput history
and final metrics using the contract in setup/EXPERIMENTS.md. Use a terse pod workload
name; the shared launcher derives it from the workspace name unless overridden.
