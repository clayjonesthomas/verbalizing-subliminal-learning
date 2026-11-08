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
   Commit and push relevant code/config. The launcher creates a detached worktree
   at that full SHA, sharing the canonical repository on the network volume.
   Never create a full clone for an individual run.
3. Generate runtime launch JSON under ignored results or outside the checkout.
   Set `source_commit` to the full SHA and `experiment` to the folder ID. The
   project execution policy supplies the canonical source and worktree paths.
   Omit `run_dir` to use `experiments/<experiment>/runs/<run-id>` on the volume.
   Never reuse an attempt's directory. If launch JSON is inside the experiment
   folder, its folder ID can be inferred. `workspace.repo` and
   `workspace.experiment` can override local workspace discovery.
4. Point `command` at the committed driver. It receives `PODRUN_DIR`, `PODRUN_ID`,
   and `PODRUN_COMMIT`; write checkpoints and large outputs under
   `PODRUN_ARTIFACT_DIR`, outside Git worktrees. Put repository-relative files
   intended for Git publication in `PODRUN_PUBLISH_DIR`.
   Write the three overall report files to `PODRUN_REPORT_DIR`.
   Verify editable dependencies resolve to the intended committed source.
5. Set finite `max_seconds` and `hook_timeout`. Reserve a dedicated pod, set
   `dedicated_pod: true` and `retire: "terminate"`. Provision/verify it through
   `setup/provision.py` and follow root AGENTS.md account, GPU and budget policy.
   `pod_name` can override the default terse workload name.
6. Configure `save` to verify/persist checkpoints and dependency metadata.
   The environment helper is the canonical
   `/workspace/projects/verbalizing-subliminal-learning/repo/setup/save.sh`;
   do not call a copied helper from a different immutable checkout path. Ensure
   the saved package inventory and lockfile match the committed experiment; do not archive venvs.
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

The default lifecycle saves, publishes and terminates after success or an explicit
stop. Unexpected failures in new budgeted runs enter a 30-minute recovery hold;
the pod-side budget watchdog enforces its deadline independently of the laptop.
The reports show the error and deadline, and bounded-hold watchers keep monitoring
until takeover or termination. Artifact/worktree publication still runs during the
hold; the worktree stays available for inspection. Cleanup failures do not extend
the original deadline. API/shared-volume outages can delay enforcement.

To repair a held run, commit/push the fix and launch a **new run ID on the same pod**,
with the same experiment and spending scope. The shared runner creates a new worktree
and atomically transfers the pod lease; old watchdogs cannot apply the old deadline
to the new worker. Keep old artifacts and provenance. Reuse completed outputs only
when their hashes and scientific compatibility have been verified. Do not run the
pod startup command again. After its publication is verified, the old worktree can
be removed with `cleanup-worktree`; artifacts remain outside it.

Set `failure_hold_seconds` to change the duration, or zero for immediate cleanup.
`--keep-on-failure` remains explicit indefinite retention, subject to the budget.
Budget exhaustion takes precedence over holds; explicit supervisor stop signals do
not initiate a recovery hold. Startup failures after the independent watchdog has
armed also get a hold; credential/guard failures before readiness require manual
inspection and must not be reported as protected. Confirm retirement via the API.

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
| `volume_id` | Project volume `orqsmyhvok`; reverify through the API |
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

## Automatic storage growth

The shared runner monitors the mounted volume from the experiment pod, independent
of the laptop watcher. At 90% usage it requests a 20% capacity increase, rounded
up to whole GB, capped at 1,000 GB. Samples run at most once per minute across
workers, plus scan time. A shared volume lock prevents duplicate growth requests.
The watcher pulls `storage.json`, including usage, capacity, resize confirmation,
and errors. Full scans can take minutes; this is preventive monitoring, not a
guarantee against a sudden large write. At the cap, the monitor reports `at_cap`
and does not increase capacity further.

## Worktree publication and cleanup

At shutdown, commit **every tracked modification/deletion and every non-ignored
new file in the run worktree**, including reports. Push and merge these changes
to `main`; an artifact branch alone does not complete publication. The runner
also publishes the repository-relative payload from `PODRUN_PUBLISH_DIR`, the
three overall reports, and artifact provenance. The recorded execution SHA stays
unchanged. Keep credentials, checkpoints and caches outside the worktree from
the start; never ignore reports to avoid publishing them.

Resolve concurrent edits and merge as much as possible, preserving both runs'
intent. The runner attempts ordinary Git merges and retries concurrent pushes;
reviewing agents must inspect substantive conflicts and resolve them when they
can. Do not stop at the first conflict or choose an entire side blindly. If a
change truly needs user review, push all saved changes to `review/<run-id>` and
mark **NOT MERGED TO MAIN**, with the branch link, in both GitHub in-progress and
final reports. The runner publishes those notices on the review branch and also
attempts to publish notices on main without changing its scientific content.
Branch protection or a remote outage can prevent main notices; the receipt records
that distinction. Never claim unmerged work was merged.

Persistent artifacts remain under `experiments/<experiment>/runs/<run-id>/artifacts`
after the run worktree is removed. `main_publication.json` records verified main
publication or the review branch and unresolved paths. Failed saves or unverified
pushes preserve the worktree. Misplaced ignored outputs also block removal, so
checkpoints cannot be silently discarded. A review branch verified on the remote
permits source-worktree cleanup; unresolved merge worktrees remain for recovery.
Use `python3 setup/experiment.py publish-main CONFIG` and `cleanup-worktree CONFIG`
after inspecting and resolving a stopped run. Historical snapshots are archives,
not candidates for automatic merging under this new-run policy.

## Activation checkpointing policy

New shared-runner launches default to `activation_checkpointing: {"backbone": false, "head": false}`. Use `podrun.activation_checkpointing.Policy.from_environment()` in new training drivers, `configure_model(model)` at setup, `assert_model(model)` after mode changes, and `run_head(loss_fn, hidden, labels)` for the output head. The runner exposes the helper through PYTHONPATH and records its effective policy in `activation_checkpointing.json`. Resumable checkpoint saving is unaffected.

The launch preflight scans pinned experiment Python for common enabling calls; do not copy checkpoint-on historical drivers into new runs. Opting in requires `reason` and `evidence` in launch config, identifying measured memory constraints or a warmed throughput benefit against fully disabled checkpointing at the same objective, BF16 precision and effective batch. See research-infra `experiment-runner/README.agents.md` for scope and limitations.
