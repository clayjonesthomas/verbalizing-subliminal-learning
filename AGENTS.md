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
cleanup retain the pod for inspection and recovery. This supersedes the older
blanket keep-alive-on-failure policy for new runs.

When retiring a pod, terminate it rather than leaving it stopped; preserve the
network volume. For subsequent work, prefer fresh capacity through
`python3 setup/provision.py` rather than restarting an old stopped pod.
Save experiment artifacts and the environment to the network volume before
terminating a usable pod. This supersedes historical stop-only workflows.

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

## Default experiment runner and watcher

Use `research-infra/experiment-runner` (the `podrun` utility) for all future
Runpod experiments in this repository. Its local checkout is normally
`~/code/research-infra`; `RESEARCH_INFRA_DIR` may override that repository path.
Use the thin entry point `python3 setup/experiment.py` from this repo. Read
`setup/EXPERIMENTS.md` before preparing a new run.

- Launch with `python3 setup/experiment.py launch CONFIG --forward-credentials`;
  append `--keep-on-failure` only when the user requests failure retention.
- Start a registered laptop watcher with `python3 setup/experiment.py watch
  WATCH_CONFIG`. Discover, stop, and archive watchers with `watchers`,
  `stop-watcher RUN_ID`, and `prune-watchers` through the same entry point.
- Keep experiment-specific drivers, settings, save/report/publish hooks, and
  outputs in `fork_experiments/<experiment>/`. Reuse the shared lifecycle and
  watcher implementation; do not create new ad-hoc supervisors, cleanup guards,
  detached shell launchers, or report polling loops unless the user requests it.
- Execute a clean, committed checkout on the persistent volume. Record its full
  SHA; keep runtime configuration/output ignored or outside that checkout.
  Use a unique persistent `run_dir` for every attempt, including retries.
- Configure the save hook to verify/persist checkpoints and preserve the Python
  environment. Configure publish/pull recovery for the final report after
  termination; a laptop can miss the last SSH sync while asleep.
- Use only the project-scoped Caleb SPAR credentials and volume checks above.
  Forward credentials through the supported memory-only channel, never files,
  command arguments, environment variables, or logs. Termination requires a
  dedicated pod; do not attach lifecycle cleanup to shared unrelated work.

This applies to future launches. Do not replace or interrupt historical runners
or active jobs merely to migrate them.

## Provisioned resources (2026-10-19)

- Original pod: `dsln6o9uc00dwc`, `verbalizing-subliminal-learning`
- Additional pod: `cgkyhjhqiwmsk7`, `verbalizing-subliminal-learning-new`
- GPU: one NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB VRAM
- Data center: `US-NE-1`
- Network volume: `ae6w0yoyy6`, 100 GB, mounted at `/workspace`

Recheck resource status through the API before relying on these details.

# Report links

When sharing HTML reports, write the full absolute filesystem path as plain
text on its own line. A clickable link may be added separately.

# Experiment provenance

Keep each fork experiment's code, configuration, logs, checkpoints, and outputs
together under `fork_experiments/<experiment>/`.
Before launching an experiment, commit all relevant code and configuration
changes. Run the committed code and record the full latest commit SHA in the
experiment metadata. Do not label a dirty working-tree run with only HEAD's SHA;
verify relevant files match the recorded commit. Apply this to future experiments
as well. Do not retain abandoned partial attempts when the user asks to delete them.

Use unique date-prefixed experiment folders with terse summary suffixes (see `fork_experiments/AGENTS.md`). The paper is cached at `references/paper/2609.16927v1.html` and `.pdf`; consult these before fetching the paper again.
