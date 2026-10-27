# Experiments: launch, watch, and report

Every new experiment uses the shared tools in `~/code/research-infra`: the pod
supervisor, registered laptop watcher, and local experiment web server. Project
entry points are `setup/experiment.py` and `setup/experiments.py`; set
`RESEARCH_INFRA_DIR` when the shared checkout lives elsewhere. Keep scientific
code here and reusable infrastructure in research-infra.

## Prepare and launch

1. Create `fork_experiments/YYYY-MM-DD-NNN-summary/` and initialize its workspace:

   ```bash
   python3 setup/experiments.py init YYYY-MM-DD-NNN-summary
   ```

   `.research-infra.json` and `setup/experiment_workspace.py` define the layout
   and templates. Initialization preserves existing reports, user notes and names.
2. Keep the driver, scientific settings, launch template, and save/publish hooks
   in that folder. Start from research-infra's `experiment-runner/examples/run.json`.
   Commit relevant code/config and create an immutable, clean checkout of its full
   SHA on the pod's network volume. The runner rejects dirty/untracked files; ignore
   generated outputs without hiding source or scientific settings.
3. Generate runtime launch JSON under ignored results or outside the checkout.
   Set `source_commit` to the full SHA, `repo` to that remote checkout, `experiment`
   to the folder ID, and `run_dir` to a unique persistent experiment-local path.
   Never reuse an attempt's directory. If launch JSON is inside the experiment
   folder, its folder ID can be inferred. `workspace.repo` and
   `workspace.experiment` can override local workspace discovery.
4. Point `command` at the committed driver. It receives `PODRUN_DIR`, `PODRUN_ID`,
   and `PODRUN_COMMIT`; write checkpoints and runtime outputs under `PODRUN_DIR`.
   Write the three overall report files to `PODRUN_REPORT_DIR`.
   Verify editable dependencies resolve to the intended committed source.
5. Set finite `max_seconds` and `hook_timeout`. Reserve a dedicated pod, set
   `dedicated_pod: true` and `retire: "terminate"`. Provision/verify it through
   `setup/provision.py` and follow root AGENTS.md account, GPU and budget policy.
   `pod_name` can override the default terse workload name.
6. Configure `save` to verify/persist checkpoints and preserve the environment.
   The environment helper is the canonical
   `/workspace/projects/verbalizing-subliminal-learning/repo/setup/save.sh`;
   do not call a copied helper from a different immutable checkout path. Ensure
   the saved environment and lockfile match the committed experiment.
7. Configure durable final publication and `workspace.pull` recovery, using absolute
   local argv paths. The shared S3 example documents the snapshot contract. A
   sleeping laptop cannot poll; the pod can terminate before its final SSH sync.
   Another running pod on the same volume can provide recovery, but requires
   running compute and is not an independent artifact store.

Launch from the laptop:

```bash
python3 setup/experiment.py launch /absolute/path/to/run.json --forward-credentials
```

The launcher initializes/registers the workspace, starts the local server and
managed watcher, and prints the stable experiment URL. Do not start a second
watcher. Check `state.json`, `guard.json` and `supervisor.log` before claiming the
experiment is ready, then give the user its URL. A printed PID only confirms process
creation. Reconcile ambiguous launches before any retry.

The default lifecycle saves, publishes and terminates after success or experiment
failure, preserving the volume. Add `--keep-on-failure` only when the user requests
failure retention. Save/report/publish failures or a lost supervisor retain the pod
for recovery. Confirm retirement through the API; a request receipt is not proof.

## Per-experiment spending limit

New named dedicated Runpod launches use the shared runner's **$50 total budget**
across all pods and attempts. Each pod runs an independent watchdog; the shared
network-volume ledger deduplicates pods and retains prior spending on retries.
At $50, workers stop and save/publish begins. At $60, remaining experiment pods
are terminated even if saving/publishing failed or hangs. This hard cutoff also
overrides `--keep-on-failure`. The network volume is preserved.

Ask the user for a specific higher total before allocating resources expected to
exceed $50. Record approval in every launch config, for example
`"budget": {"limit_usd": 100, "approval": "User approved $100 for this experiment"}`.
Choose the budget before the first launch; do not change an experiment ID or erase
its `budget_policy.json` / shared ledger to reset spending. The cutoff is always
120% of the approved amount; that margin is for cleanup, not additional training.

Always use `--forward-credentials`. Verify `state.json.budget_watchdog.ready` and
`budget.json` in the run snapshot, alongside supervisor readiness. The initial
watchdog must arm successfully before scientific work starts. Setup/idle time is
included from the API pod creation timestamp, using the same GPU + container-disk
cost basis as the report graph. Shared-volume storage remains excluded.

No local deadline fallback is configured. Shared-volume outages, provider API
outages, or loss of every watchdog can delay/prevent termination. Provisioning
before the watchdog starts is not protected: arm promptly, and retire pods if
launch fails. No existing job is automatically migrated. To adopt an old experiment,
first retire its unguarded pods and reconcile their spend; supply that retired-pod
amount as `budget.prior_spend_usd` (exclude the new pods). Missing legacy baselines
are rejected. `budget_policy.json` is provenance and should be committed.

## Account and volume scope

Both remote `provider` and local `laptop_provider` must identify the same dedicated
pod, verified volume and mount, team and account:

| Field | Project scope |
| --- | --- |
| `team_id` | `cmu31hyiy002gk9s0xik85gob` |
| `account_id` | `user_3JNPrVMTYwugXaYJytSv2t8lRsv` |
| `volume_id` | Project volume `ae6w0yoyy6`; reverify through the API |
| `mount_path` | `/workspace` |
| `pod_id` | Actual dedicated pod ID discovered through the API |
| Laptop `key_file` | `~/.config/runpod/accounts/verbalizing-subliminal-learning/api_key` |

`--forward-credentials` sends the scoped Authorization header through SSH/stdin
into supervisor memory; the remote provider needs no key file. Never place credentials
in files, command arguments, environment dumps, reports or logs. See root AGENTS.md
for identity checks and GPU allocation policy.

## HTML reports and research notes

Use the experiment URL as the report interface. The server discovers existing
experiments and gives each one four views:

| File | Purpose |
| --- | --- |
| `research_notes.html` | User-owned WYSIWYG notes; preserve formatting and content |
| `in_progress.html` | Current findings, intermediate measurements and Plotly graphs |
| `final_report.html` | Final findings, behavior evaluation, implications and limitations |
| `methods.html` | Hypotheses, controls, methods, scientific config and evaluation definitions |

Reports are standalone HTML with inline styles and Plotly assets, viewable without
this server. The tabs display these files directly and expand to their full height.
The default renderer updates files carrying `<!-- expweb:generated v1 -->` only.
Remove that marker when replacing a generated report with a custom scientific report.
Keep results pending until measured; keep methods/config details in their own tab.
Preserve user notes and the display name. The browser name editor changes the saved
name and browser title, leaving folder IDs and URLs stable. Edits do not auto-commit.
Legacy Markdown is preserved, with user notes copied to HTML on first use.

Managed local snapshots and generated watch configuration live under ignored
`results/podrun/<run_id>/`. Execution IDs and telemetry directories are internal
provenance/cleanup bookkeeping. There is no run selector or separate report set.
Drivers and agents write only the three overall HTML reports to `PODRUN_REPORT_DIR`,
which is the experiment folder in the remote checkout. The watcher copies those
files to the local experiment root. Keep user research notes local and untouched.

Append `metrics.jsonl`, `training.jsonl` or `evaluations.jsonl` records under
`PODRUN_DIR` with numeric `step`, `loss`/`nll`, and throughput fields. Write current
progress atomically to `progress.json`. The runner provides GPU utilization and
VRAM history. Put final behavior scores and sample counts in `final_report.html`. Structured
`final.json` or `summary.json` is optional input for the default report renderer. Scientific interpretation and evaluation quality remain the driver's
and agent's responsibility.

The UI refreshes every 30 seconds and swaps changed HTML only after background
loading and layout. Homepage activity requires a fresh active runner heartbeat;
stale/disconnected/unclassified records show Unknown.

## Local services and recovery

```bash
python3 setup/experiments.py start
python3 setup/experiments.py status
python3 setup/experiments.py stop
python3 setup/experiment.py watchers
python3 setup/experiment.py stop-watcher RUN_ID
python3 setup/experiment.py prune-watchers
```

The default server is `http://127.0.0.1:8765`. Its process registry is
`~/.cache/research-infra/experiment-browser/`; watcher registration, heartbeat and
logs live under `~/.cache/research-infra/podrun/`. Ownership locks protect against
stale PIDs. Watchers have bounded lifetimes and exit on terminal run states.
Stopping the web server or watcher never stops the pod. Pruning archives inactive
watcher registrations and preserves logs. Never delete live registries or signal a
process based only on a stale PID file.

For watcher recovery, use `python3 setup/experiment.py watch WATCH_CONFIG` with the
generated config under that run's local snapshot directory. If launch succeeded but
watcher startup failed, restart only the watcher. Existing jobs can keep their
historical supervisors and layouts; do not migrate or interrupt them just for a tool
upgrade. Use `setup/README.md` for environment inspection and startup repair.

## Local time and cost

`.research-infra.json` fixes the display timezone to America/Los_Angeles. The
launcher passes it as `PODRUN_TIMEZONE` for report generation on the pod. Use
shared graphing helpers for local timestamps, time axes and hover values; retain
machine epochs only in internal telemetry.

The server automatically prepends a Plotly cumulative-cost graph to the overall
in-progress HTML. It uses rr runpod-spend's durable ledger and scoped read-only
refresh, sums assigned pods without counting retries twice, and includes GPU plus
container storage. Shared network storage is excluded. Billing evidence, freshness
and missing charges are labeled, and a snapshot is saved to results/experiment_cost.json.
Authors continue to write only the three overall HTML files; the server maintains
its own cost/time regions.
