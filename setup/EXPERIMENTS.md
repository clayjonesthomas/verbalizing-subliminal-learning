# Default Runpod experiment workflow

Use the shared runner in `~/code/research-infra/experiment-runner` for new
Runpod experiments. `python3 setup/experiment.py` delegates directly to that
checkout; set `RESEARCH_INFRA_DIR` if the research-infra repository lives
elsewhere. No package installation or copy of its supervisor is required here.
Run this entry point on the laptop; launch deploys the shared runner bundle to
the pod. Existing experiment drivers and historical jobs remain valid.

## Prepare a run

1. Keep the experiment driver, training settings, launch/watch templates, and
   save/report/publish hooks in a unique `fork_experiments/YYYY-MM-DD-NNN-summary/`
   directory. Copy the shared runner's `examples/run.json` and
   `examples/watch.json` as starting templates; fill in this project's scope below.
2. Commit all relevant code and settings. Prepare a clean immutable checkout of
   that full commit SHA on the pod's network volume. The runner rejects a dirty
   checkout, including untracked files. Use an experiment-local `.gitignore` for
   generated results, report HTML, and runtime manifests; do not hide source or
   scientific settings. Do not commit unrelated working-tree changes.
3. Generate the final launch JSON after committing, under ignored results or
   outside the checkout. Set `source_commit` to the full SHA. Keep the committed
   template and scientific settings as the provenance inputs; this avoids trying
   to put a commit's own SHA inside that commit.
4. Set `repo` to the clean remote checkout and `run_dir` to a unique persistent
   `fork_experiments/<experiment>/results/<run-name>/` location on `/workspace`.
   Use a different directory for every attempt. Point `command` at the committed
   experiment driver, e.g. `["bash", "setup/run.sh", "python", "-m", "fork_experiments.<alias>.run"]`.
   Run modules from that checkout and verify local editable dependencies resolve
   to committed source rather than a different mutable checkout.
   The driver receives `PODRUN_DIR`, `PODRUN_ID`, and `PODRUN_COMMIT`; write
   checkpoints, logs, and atomic `progress.json` updates under `PODRUN_DIR`.
5. Configure `save` to verify/persist experiment checkpoints and archive the
   environment before retirement. The existing environment helper is
   `/workspace/projects/verbalizing-subliminal-learning/repo/setup/save.sh`.
   It is bound to that canonical checkout; do not call a copied `setup/save.sh`
   from an immutable checkout at a different path. An experiment-specific save
   hook can verify checkpoints, call the canonical helper, and flush its run
   directory. Ensure the environment/lockfile match the committed experiment.
6. Set a finite `max_seconds` and suitable `hook_timeout`. Set `dedicated_pod`
   to true and `retire` to `terminate` only on a pod reserved for this job.
   Provision and verify SSH/Python/CUDA first using the existing setup helpers.
   Prefer RTX PRO 6000; announce the model/rate before using a more expensive
   fallback. Preserve the existing <$5/hour GPU limit unless explicitly changed.
7. Configure a durable final snapshot publisher and the watcher's matching `pull`
   command. The shared S3 example shows the snapshot contract; use an already
   configured artifact destination. Another running pod on the same volume can
   provide read-only recovery, but is not independent of running compute. Without
   recovery, a sleeping laptop can miss the final SSH sync before termination.

Both the pod's `provider` and the laptop's `laptop_provider` must name the same
pod, team, account, network volume, and mount path:

| Field | Project scope |
| --- | --- |
| `team_id` | `cmu31hyiy002gk9s0xik85gob` |
| `account_id` | `user_3JNPrVMTYwugXaYJytSv2t8lRsv` |
| `volume_id` | `ae6w0yoyy6` (reverify through the API) |
| `mount_path` | `/workspace` |
| `pod_id` | Actual dedicated pod ID discovered through the API |
| Laptop `key_file` | `~/.config/runpod/accounts/verbalizing-subliminal-learning/api_key` |

With `--forward-credentials`, the pod's provider needs no `key_file`: the scoped
Authorization header travels over SSH and anonymous stdin pipes into supervisor
memory. Never put the key in experiment files, environment variables, argv or logs.
Use the same laptop provider scope in the watch config's `provider` field.

## Launch and watch from the laptop

From this repository root, substitute the generated configuration paths:

```bash
python3 setup/experiment.py launch /absolute/path/to/run.json --forward-credentials
python3 setup/experiment.py watch /absolute/path/to/watch.json
python3 setup/experiment.py watchers
```

If the user asks to keep a pod after experiment failure, append
`--keep-on-failure` to launch. Otherwise the runner saves and terminates after
success or experiment failure. Save/report/publish failures or a lost supervisor
retain the pod; inspect and recover it. Network volumes are preserved. Confirm
retirement through the API: `termination_requested` is intent, not confirmation.

Set watch `destination` to a local run-specific directory, preferably
`fork_experiments/<experiment>/results/<run-name>/lifecycle/`, so its managed
`report.html` stays with the experiment and does not overwrite the top-level
scientific report. The watcher mirrors the runner snapshot, not arbitrary analysis
files. Publish/sync additional report artifacts explicitly if the experiment needs
them, preserving the five-bullet overview convention for scientific reports.

Launch prints the detached guard PID, not a readiness guarantee. Check
`state.json`, `guard.json`, and `supervisor.log` for successful preflight. Do not
retry an ambiguous launch with the same run directory; reconcile it first.

## Manage watcher processes

```bash
python3 setup/experiment.py watchers
python3 setup/experiment.py stop-watcher RUN_ID
python3 setup/experiment.py prune-watchers
```

The registry is `~/.cache/research-infra/podrun/`. Watchers use ownership locks,
record their status and heartbeat, have bounded lifetimes, and exit after terminal
run states. Stopping a watcher does not stop the experiment. Pruning archives
inactive registrations and preserves their logs. Never delete the registry while
watchers are alive, or signal a process based only on a stale PID file.
Use absolute paths for local script/config arguments in `pull`, since the shared
watcher runs from the research-infra utility directory.

`setup/provision.py` remains the capacity/startup helper and `setup/runpod.py`
remains the scoped access helper. Do not use their historical save-and-stop
workflow as the default experiment lifecycle. Do not create new ad-hoc
supervisors, cleanup guards, or polling loops in experiment folders; add reusable
lifecycle features in research-infra and keep scientific logic here.
