# Failure-only Codex wake-ups

New `setup/experiment.py run` / `submit` commands launched by Codex automatically
register `CODEX_THREAD_ID` with the shared research-infra dispatcher. One laptop
process checks the global pool, regardless of experiment count. Existing per-run
report watchers are separate and stay unchanged. Healthy/successful checks use no
model tokens. Failures queue a message in the launching conversation.

The dispatcher reads the Caleb SPAR volume through signed S3 GETs with explicit
team/account identity and the repository `.env` `S3_API_KEY`. No GPU is provisioned
for monitoring. Runpod controller, task, missing-credential and ambiguous-allocation
failures are actionable; ordinary capacity waits are not. Uncaught exceptions in
the detached pool bootstrap also create local failure receipts. Existing legacy
workers are not migrated or automatically monitored by this new pool dispatcher.

Start from an ordinary laptop shell with access to the running Codex daemon:

```sh
python3 setup/failure_watch.py start
python3 setup/failure_watch.py status
python3 setup/failure_watch.py stop
```

Start verifies daemon access. If an agent sandbox denies the socket, authorize the
service launch outside that sandbox. Future submissions reuse the running singleton.
After reboot, launch registration starts it again; no login service is installed.

To associate an existing **pool** experiment with its original agent:

```sh
python3 setup/failure_watch.py register --experiment ID --thread THREAD_UUID
```

For delegated work that must report back to another conversation, explicitly add
`--notify-thread PARENT_UUID`, or set `PODRUN_NOTIFY_THREAD_ID=PARENT_UUID` when
submitting. The watcher includes the parent UUID and required handoff directly in
the agent's wake-up message; it does not depend on rereading edited instruction
files. No extra recipient is inferred for ordinary experiments.

To retest just the handoff with a saved failure receipt (no new GPU allocation):

```sh
python3 setup/failure_watch.py replay --receipt /absolute/saved-receipt.json \
  --thread TEST_WORKER_UUID --notify-thread PARENT_UUID
```

An explicit registration alerts on current failures too. Do not register archived
experiments unless you want their failures investigated. Registration without
`--thread` uses `CODEX_THREAD_ID`; human shell launches require the explicit UUID.
To disable an experiment or deliberately transfer its owner, first use:

```sh
python3 setup/failure_watch.py unregister --experiment ID
```

Durable registrations, status and delivery receipts live under
`~/.cache/research-infra/failure-watch/`. A failure is queued once per attempt.
`queued_wake_unconfirmed`, `uncertain`, and `dispatching` require inspecting the
thread before manually reconciling delivery; they are never blindly requeued.
Success means the CLI accepted the queue/resume request, not that repair finished.
Polling resumes after laptop wake; recovery and budget deadlines continue on pods.
S3 outages appear in dispatcher status and cannot be mistaken for successful reads.

See research-infra `experiment-runner/FAILURE-WATCH.md` for lifecycle and transport
details. The dispatcher never retries experiments itself. The notified agent must
diagnose, commit and validate a causal repair and preserve the existing budget.
