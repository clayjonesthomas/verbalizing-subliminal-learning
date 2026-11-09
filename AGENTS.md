# Runpod account for this repository

Use only the Runpod API or MCP. All pods and volumes for this repository must
belong to the **Caleb SPAR** team, never the Clay personal account.

- Read the API key from
  `~/.config/runpod/accounts/verbalizing-subliminal-learning/api_key` directly
  into the client's `Authorization: Bearer ...` header. Never print or copy it
  into this repository or logs.
- Send `X-Team-Id: cmu31hyiy002gk9s0xik85gob` on every API request. This key
  otherwise defaults to the Clay personal account.
- Before creating resources, query GraphQL `myself { id email }` with both
  headers. Require account ID `user_3JNPrVMTYwugXaYJytSv2t8lRsv`
  (`caleb.d.biddulph+spar@gmail.com`). Fail closed on a mismatch.
- Do not fall back to `~/.config/runpod/api_key`, ambient credentials, or an
  MCP connection whose team scope has not been verified. Do not change global
  Runpod settings.
- GPU capacity is scarce. Query live GPU availability, create the network
  volume in the available GPU's data center, and immediately create the pod
  with that volume attached. If capacity disappears, remove only the newly
  created empty volume from that unsuccessful attempt before trying elsewhere.
- Discover public IPs and SSH ports through the API; for REST v1, the direct
  SSH port is `portMappings["22"]`.

## Network-volume capacity

The current project volume is `orqsmyhvok` in US-NE-1 (300 GB at migration).
The shared runner enables pod-side storage monitoring for dedicated experiments:
at 90% usage, grow by 20%, rounded up to whole GB, with a hard cap of 2,000 GB (user approved 2026-11-08).
Workers coordinate through a shared volume lock; the watcher pulls `storage.json`.
Do not assume `df` reports the volume quota: use measured directory usage and the
Runpod API capacity. At the cap, request cleanup or a new user-approved policy.

## Creating additional pods

For new budgeted experiments, the shared runner defaults to a 30-minute recovery
hold after unexpected worker/startup failures. Save and publish the failure, keep
the pod and its worktree for a quick fix, and show the recovery deadline. The
independent pod-side budget watchdog terminates the pod if the hold expires.
Successful runs and explicit stop/termination requests still retire promptly.
Budget limits override recovery holds; shared-volume/API outages can delay enforcement.

For a quick fix during the hold, commit and push corrected code, then launch a new
run ID/worktree on the same verified pod and same experiment budget. Do not restart
setup/start.sh, reuse the old run directory, change the old execution SHA, or provision
a replacement unnecessarily. The new supervisor must acquire the pod lock and claim
its recovery lease before work starts; an expired/terminating lease rejects takeover.
Preserve old artifacts and lifecycle records; verified artifacts may be reused.
`failure_hold_seconds: 0` explicitly disables the default hold. Use
`--keep-on-failure` only when the user asks for indefinite inspection retention
(the experiment budget still applies). Before watchdog readiness, failed launches
remain a manual-recovery case: inspect the pod; never claim an armed recovery timer.

When retiring a pod, terminate it rather than leaving it stopped; preserve the
network volume. For subsequent work, prefer fresh capacity through
`python3 setup/provision.py` rather than restarting an old stopped pod.
Save experiment artifacts and dependency metadata to the network volume before
terminating a usable pod.

Do not save or restore full Python environment archives by default. Fresh pods
install from the committed lockfile; experiment-specific dependencies need their
own pinned install recipe. Preserve lockfiles/package inventories, not venv tar
files. Keep shared model/dataset caches. Historical experiment scripts are not
retroactively migrated; some explicitly require archives that have been purged.

For another pod on the existing volume, run `python3 setup/provision.py`.
Allow at most two concurrent Runpod provisioning processes (user approved 2026-11-08), shared across CPU and GPU allocation. Each holds one slot through identity/catalog checks, allocation retries, and readiness checks. Additional invocations wait without contacting Runpod. The original `provision.lock` remains slot one so an already-running provisioner still counts; slot two is `provision.lock.slot2`. Shared API rate limiting remains active. This does not limit running workloads.
Prefer full NVIDIA RTX PRO 6000 GPUs (96 GB) over MIG slices unless the user
specifies another GPU. The four-rule training comparison measured about 46 minutes
on a full GPU versus 100 minutes on a 48 GB MIG slice for 2,500 steps.
Only NVIDIA RTX PRO 6000 variants may be provisioned. Do not include H100,
H200, A100, or other GPU families in placement requests or fallback searches.
Keep all eligible RTX PRO 6000 variants in every retry cycle, preferring the
full 96 GB Server and Workstation variants before MIG slices. If capacity is
unavailable, keep retrying this family; do not fall back to other models.
It requires at least 48 GB VRAM and a catalog
GPU price strictly below $5/hour (storage excluded), in US-NE-1.
Availability can report NONE even shortly before a successful placement; keep
retrying explicit capacity rejections until successful or the user stops you.
Never retry an ambiguous creation outcome without reconciling the account.

Always configure `setup/start.sh` as the new pod's startup command and verify
SSH plus the project Python/CUDA environment before reporting it ready. On an
already-created pod without that command, run `setup/boot.sh` over direct SSH,
then `setup/runpod.py --pod ID enable-startup` and verify readiness again.
Do not run `/start.sh` manually inside an already-running SSH session.

## Experiment runner and local web workspace

Use the shared `~/code/research-infra` runner, watcher, and experiment web server
for every new experiment. `RESEARCH_INFRA_DIR` can override that checkout.
Read `setup/EXPERIMENTS.md` for the configuration and metrics contract.

1. Create the date-prefixed experiment folder and run
   `python3 setup/experiments.py init EXPERIMENT_ID` before committing its code/config.
2. Launch from the laptop with `python3 setup/experiment.py launch CONFIG
   --forward-credentials`. The launcher initializes/registers the workspace, starts
   the local server and managed watcher, and assigns a terse pod workload name.
   Do not start a duplicate watcher or create ad-hoc lifecycle/polling scripts.
3. Verify remote supervisor readiness separately, then give the user the stable
   experiment URL printed by launch. Use the server as the primary report interface.
4. Keep agent reports in standalone `in_progress.html`, `final_report.html`, and
   `methods.html` at the experiment root (`PODRUN_REPORT_DIR` on the pod), with
   inline Plotly assets. There are no separate per-run report sets. Preserve the
   user's WYSIWYG
   `research_notes.html` and display name. Keep hypotheses, methods/config and
   evaluation definitions out of the progress/final reports.

`python3 setup/experiments.py start|status|stop` manages the local web server.
`python3 setup/experiment.py watchers`, `stop-watcher RUN_ID`, and `prune-watchers`
manage watchers. Stopping the server or a watcher does not stop an experiment.
Use manual `watch CONFIG` only to recover a watcher or support an existing run.
Configure durable publish/pull recovery so final results survive laptop sleep and
pod termination. Implement reusable infrastructure in research-infra, scientific
logic in the experiment folder. Do not migrate or interrupt active jobs solely to
adopt a new infrastructure version. Use this repo's existing local server for UI
verification; restart it after backend changes instead of spawning extra test servers.

# Report tracking

Never gitignore report files, including generated HTML reports, Markdown reports,
methods, and research notes. Keep reports eligible for version control, including
reports inside otherwise ignored output directories. Commit reports when publishing
experiment results to GitHub.

# Report links

When sharing a GitHub URL for an HTML report, prefix the full GitHub file URL
with `https://html-preview.github.io/?url=` so the link opens a rendered preview.

Share the experiment server URL as the primary report entry point. When sharing
an HTML file directly, also write its full absolute filesystem path as plain text
on its own line; a clickable file link may be added separately.

# Experiment provenance

Keep each fork experiment's code, configuration, logs, checkpoints, and outputs
together under `fork_experiments/<experiment>/`.
Before launching an experiment, commit all relevant code and configuration
changes. Run the committed code and record the full latest commit SHA in the
experiment metadata. Do not label a dirty working-tree run with only HEAD's SHA;
verify relevant files match the recorded commit. Apply this to future experiments
as well. Do not retain abandoned partial attempts when the user asks to delete them.

Use unique date-prefixed experiment folders with terse summary suffixes (see `fork_experiments/AGENTS.md`). The paper is cached at `references/paper/2609.16927v1.html` and `.pdf`; consult these before fetching the paper again.

# Time and cost reporting

## Network failure diagnosis

Network access is enabled in this workspace. A connectivity error such as
`Network is unreachable` is not evidence that elevated permissions are needed.
Diagnose connectivity separately from filesystem or sandbox permissions; do not
describe elevation as a confirmed fix without evidence. Existing experiment
authorization remains valid after a transient connection failure. If a higher-
priority tool policy requires an escalation attempt, explain that policy as the
reason for the approval request rather than claiming a demonstrated network limit.

## Experiment spending limit

The default spending limit is **$50 total per experiment**, summed across all
pods, setup/idle time, retries, and replacement pods. It is not a per-pod or
per-seed allowance. Ask the user for a specific larger total before allocating
compute if the experiment is expected to need more. Never silently raise or
reset the limit by starting another attempt.

New experiments must have an enforced cost shutoff independent of the laptop
and report server. Run a separate watchdog on each experiment pod, coordinating
the experiment-wide budget through the shared network volume. Begin graceful
shutdown at the approved budget; force termination at 120% of that budget if
cleanup has not finished, even if publication is unsuccessful. Preserve artifacts
on the network volume. There is no local fallback deadline; shared-volume outages may delay enforcement.
Use the shared runner and the report billing scope; the 20% margin is for shutdown,
not extra training.
Do not claim this policy is an implemented safeguard: verify enforcement is
available and tested before launching. If it is unavailable, implement it first.

Use America/Los_Angeles local time (PDT/PST as appropriate) for displayed timestamps
and wall-clock plot axes. Never present Unix seconds or UTC as the report clock.
Use research-infra graphing helpers; the runner sets PODRUN_TIMEZONE on the pod.
The web server adds a cumulative cost graph at the top of in_progress.html from
the same scoped billing ledger as rr runpod-spend. Keep writing only the three
overall reports; do not create a separate cost report or manually estimate charges.

GPU utilization, cost over time, unclipped gradient norm, and learning rate graphs belong only in
the in-progress server report, never the final report.

## Shared graphing enforcement for new experiments

Use `research_graphing.chart_html`, `chart_grid`, and `plotly_script` for all
new report charts. Do not export figures directly through Plotly `to_html` or
`write_html`. Shared helpers own sizing, two-column layout, styles, legends,
interaction settings, and offline assets. Put reusable improvements in
research-infra rather than copying chart wrappers into experiments.

New workspace initialization creates `report_policy.json`; the project launcher
checks opted-in report source and HTML before launch. Do not delete or disable
that policy to bypass failures. Render representative data and inspect the
existing server report before launch. Existing experiments remain exempt; do
not migrate old reports unless explicitly requested.


## New-run Git publication and artifacts

Use the shared runner's managed detached worktree for each new run; do not clone
one repository per experiment. Store checkpoints, logs and large outputs in its
persistent off-repo `PODRUN_ARTIFACT_DIR`, with provenance tying them to the run.
Use `PODRUN_PUBLISH_DIR` for repository-relative output files intended for Git.
Artifacts stay on the network volume after worktree cleanup.

Shutdown must commit and push **all tracked changes/deletions and all non-ignored
new files in the isolated run worktree**, including reports, and merge them to
`main`. A selected-file artifact branch is not sufficient. Resolve conflicts as
best you can, preserving both changes' intent. If a change truly needs user review,
push it to a separate review branch and prominently mark **NOT MERGED TO MAIN**,
with its branch link, in both final and in-progress reports as published on GitHub.
Publish the notice to main's report pages as well when permitted. Do not silently
leave changes local or claim an unmerged branch is merged. Never force-push.
Verified publication to main or an explicitly flagged review branch permits
worktree removal; unverified pushes and misplaced ignored outputs block removal.
This policy is for new runs, not permission to merge archived historical checkouts.

## Training throughput memory (2026-11-07 benchmark)

Measured in `fork_experiments/2026-11-07-002-lls-speed/results/lls-speed-a1/case-*.json` on Qwen3.5-9B / RTX PRO 6000, BF16. At microbatch 2, disabling backbone and LM-head activation checkpointing reduced student DPO update time 9.68 -> 7.17 s and direct-soft DPO 2.99 -> 2.24 s (about 25-26% less time). Peak allocation rose to 31.1 / 28.7 GiB. Larger microbatches were slower and had greater BF16 gradient differences; do not assume larger is faster. Benchmark representative lengths before transferring this choice to SFT or long transcripts. Head checkpointing is also OFF by default; enable it only after a measured memory failure or demonstrated throughput benefit (see the activation-checkpointing policy below). Preserve dtype, objective, effective batch, example order, optimizer schedule, and loss normalization. Check loss/gradient agreement; these changes are not bitwise identical in BF16. Do not use FP8/quantization as a throughput shortcut.

For DPO only, cache frozen reference log-probabilities using exact model/tokenizer/software/token and batch-layout identity. The benchmark measured 4.92 / 1.85 s with warm caches, excluding cache construction; do not report this as end-to-end speedup. Policy scores must remain fresh. Persist/resume optimizer, RNG, sampler position and selection history; saved checkpoints are independent of activation checkpointing. Use independent GPU workers across experiments/stages, shared immutable caches, and release GPUs when their queues finish.


# Activation checkpointing: opt-in only

Default **both backbone and LM/output-head activation checkpointing OFF** for new experiments. This is the user's explicit preference (2026-11-08). Do not confuse activation recomputation with saving resumable checkpoints; continue saving model, optimizer, RNG and sampler state.

Use the shared podrun activation-checkpointing policy/helper and record the effective settings. Benchmark the fully disabled setting first on representative and longest batches at the intended BF16 precision and effective batch. Enable checkpointing only for a measured memory constraint or a measured net throughput win, with the reason and evidence recorded in launch configuration. A percentile/length heuristic, GPU utilization, or an assumed safety margin is not evidence. Test backbone and head independently; do not silently retain head checkpointing. Preserve the objective, precision, optimizer schedule and effective batches, and verify loss/gradient agreement. Do not copy older checkpoint-on scripts without adapting them. Report exceptions explicitly. Do not change or restart an active pinned run merely to adopt this default.

## Approved GPU comparison exception (2026-11-08)

The user explicitly requested a representative Qwen3.5-9B SFT/DPO GPU pilot,
including alternatives to RTX PRO 6000. For `2026-11-08-002-gpu-efficiency` only,
exact GPU selections may include A100 80 GB PCIe/SXM, H100 80 GB PCIe/SXM, L40S,
and H200, alongside the RTX baseline. Use `setup/provision.py --allow-non-rtx
--gpu EXACT_ID --max-hourly 6` for these scoped allocations (H200 catalog rate
$5.29/hour was disclosed before the request). The pilot uses the default $50
total allowance, with $40 graceful / $48 hard GPU cutoffs. Account, volume,
precision and checkpointing policies remain in force. This does not change
the default GPU family for other experiments.

# Default Git publication

Commit, push, and merge completed task changes into `main` by default. Do not stop at a feature or artifact branch unless the user explicitly asks, branch protection blocks the merge, or an unresolved conflict requires their input. Preserve unrelated local changes and never force-push.

## Cross-region pilot storage exception (2026-11-08)

For `2026-11-08-002-gpu-efficiency`, the user explicitly authorized other data centers with pod-local storage and no new network volumes. Use `--pod-local-gb 120` with exact GPU IDs, then the shared-runner-budgeted `setup/bootstrap_local_pod.py` before scientific launch. The temporary empty-disk pod starts image SSH services; bootstrap copies required inputs, verifies project Python/CUDA, and configures `setup/start.sh` for future starts. Publish all outputs externally before terminating these pods. The committed budget partition bounds the aggregate pilot, including prior spend and each allocation's hard cutoff, below $50; never reset an allocation on retries.

## Cached soft-prompt generation (2026-11-08)

Default soft-prompt verbalization and beam-search continuation generation to KV/state
caching. Keep an explicit uncached option for reproduction and correctness pilots.
Use a fresh model-native cache per continuation; preserve temperature, stop tokens,
precision, and the search objective. Benchmark speed and peak allocated/reserved
GPU memory and check cached/uncached logits before adopting on a new architecture.
Scoring minibatch size is independent: benchmark it, but do not change an active
run's minibatch without user authorization. This preference does not enable
activation checkpointing; backbone and output-head recomputation remain off.
