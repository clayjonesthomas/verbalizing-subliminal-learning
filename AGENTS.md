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

When retiring a pod, terminate it rather than leaving it stopped; preserve the
network volume. For subsequent work, prefer fresh capacity through
`python3 setup/provision.py` rather than restarting an old stopped pod.
Save experiment artifacts and the environment to the network volume before
terminating a usable pod. This supersedes historical stop-only workflows.

For another pod on the existing volume, run `python3 setup/provision.py`.
It retries specific NVIDIA GPU models with at least 48 GB VRAM and a catalog
GPU price strictly below $5/hour (storage excluded), cheapest first, in US-NE-1.
Availability can report NONE even shortly before a successful placement; keep
retrying explicit capacity rejections until successful or the user stops you.
Never retry an ambiguous creation outcome without reconciling the account.

Always configure `setup/start.sh` as the new pod's startup command and verify
SSH plus the project Python/CUDA environment before reporting it ready. On an
already-created pod without that command, run `setup/boot.sh` over direct SSH,
then `setup/runpod.py --pod ID enable-startup` and verify readiness again.
Do not run `/start.sh` manually inside an already-running SSH session.

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
