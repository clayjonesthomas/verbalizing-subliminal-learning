# Figure 4 greedy comparison — awaiting review

No replacement run has been launched and no pod has been allocated.

## Author-facing experiment

`experiment_scripts/greedy_comparison/run.sh` consumes existing SALVE results,
runs `recover_and_evaluate.py --greedy-from`, and calls the original
`optimizer_comparison/score_fluency.py` for greedy and SALVE prompts.

The original prompted-learning `run.sh` is unchanged. There is no separate
`greedy_readout.py`. Greedy output follows `optimizer_comparison/run_method.py`:
`<animal>/greedy/seed<seed>/result.json`, plus the shared
`prompt_fluency.json`.

## Local experiment support

- `run_figure4_salve.sh ANIMAL` creates five control outputs on one pod. Four dedicated pods run cat, dog, eagle, and owl concurrently. This helper is support-only.
- `run.py --stage salve --animal ANIMAL` calls that worker’s helper.
- `run.py --stage greedy --control-plan PATH` verifies all four workers and all 20 checkpoints, then calls the author-facing script. `--salve-runs PATH` still accepts existing author outputs.
- `parallel.py` generates four distinct SALVE runner configs and `control-plan.json` from `run.salve.template.json`; `run.greedy.template.json` consumes that plan.
- `report.py` joins SALVE and optimizer-style greedy results; `report.css` controls presentation.
- `hooks.py` saves the environment and publishes results through the shared runner.
- `methods.html` holds the protocol and metric definitions.

Do not provision or launch until user review is complete.

## Four-pod preparation (after approval)

1. Provision four dedicated pods through the existing team-scoped setup tooling.
2. Generate configs with `parallel.py --pod-ids CAT_POD DOG_POD EAGLE_POD OWL_POD --remote-root REMOTE_ATTEMPT_ROOT --batch UNIQUE_SUFFIX --out NEW_LOCAL_CONFIG_DIR`. Use a unique attempt path under this experiment’s `results/` on `/workspace`.
3. Deploy the same committed checkout to each config’s `repo` and copy `control-plan.json` to its recorded `remote_plan` path. Fill the existing durable publication/pull recovery configuration.
4. Launch each of the four configs using `python3 setup/experiment.py launch CONFIG --forward-credentials`. These launch detached supervisors; do not wait for one experiment to finish before launching the others. Use their managed watchers only.
5. Each worker saves/publishes its own artifacts and terminates its own pod. The shared network volume preserves the checkpoints.
6. Once all four workers finish, the separate greedy stage consumes the control plan. Missing results, empty checkpoints, wrong animal/seed assignments, and mismatched commits block it before any readout.

The four workers publish data, not four competing report sets. The greedy stage
renders the single experiment-wide comparison from the combined controls.
The plan generator does not allocate, deploy, launch, or poll anything.
