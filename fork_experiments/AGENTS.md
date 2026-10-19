# Fork experiment layout

Keep each new experiment in its own `fork_experiments/<experiment>/` directory.
Store its runner, experiment-specific code, configuration, notes, logs, and
results together there. Use `results/<run-name>/` for individual runs, rather
than the top-level `outputs/` directory. Mirror this layout on Runpod.

Reuse shared library code and shared model/dataset caches; keep modifications
specific to an experiment inside its directory rather than changing the paper's
reference implementation. Do not copy secrets into experiment directories.

Before every experiment launch, commit all relevant code/configuration changes,
verify the executed files match that commit, and record its full SHA in the run
metadata. Generated results do not need to be committed before a subsequent run,
but must not conceal uncommitted changes to executable code or configuration.
