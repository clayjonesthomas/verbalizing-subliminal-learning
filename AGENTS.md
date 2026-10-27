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

## Creating additional pods

For new experiments, use the shared runner's default cleanup policy: save
artifacts and the environment, then terminate the dedicated pod after success
or experiment failure. If the user asks to retain failures for inspection, use
`--keep-on-failure`. Save/publish failures, lost supervisors, and incomplete
cleanup retain the pod for inspection and recovery.

When retiring a pod, terminate it rather than leaving it stopped; preserve the
network volume. For subsequent work, prefer fresh capacity through
`python3 setup/provision.py` rather than restarting an old stopped pod.
Save experiment artifacts and the environment to the network volume before
terminating a usable pod.

For another pod on the existing volume, run `python3 setup/provision.py`.
Prefer NVIDIA RTX PRO 6000 capacity by default unless the user specifies another
GPU, including slices meeting the job's VRAM requirements. If that capacity is
unavailable, a more expensive eligible GPU is permitted, but tell the user its
model and hourly GPU rate before allocating it. `setup/provision.py` tries the
preferred family first; if it reports fallback options, announce the fallback
and rerun with `--allow-fallback`. It requires at least 48 GB VRAM and a catalog
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
