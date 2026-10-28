# Four-rule recovery: ten seeds

Review-only reconstruction of the stopped ten-seed experiment. No runs were launched
from this branch. Scientific entry points are in `experiment_scripts/four_rules/`;
this folder only adapts them to the shared research-infra runner and reports.
The original `salve/` implementation and dependency lockfile are unchanged.

## Prerequisites and provenance

Use the existing canonical project environment/provisioning checkout described in
`setup/EXPERIMENTS.md`. This branch includes the thin runner/browser entry points;
it does not vendor the shared supervisor or replace the machine's provisioning
configuration. Shared infrastructure must include commit
`f04e225e80796f935d31ab19588138f3a75d6c0d` (budget watchdog, cleanup backoff and error
logging). Use its tested installation recipe and the original pinned `uv.lock`.
Only the Caleb SPAR account and project volume are allowed.

The immutable historical execution remains on `experiments/four-rule-ten-seeds`
(source `5b87f2d515c0437ec7b263eb65a3c0707bd2d350`). Its five artifact branches are
`artifacts/rules-ten-w0-a1` through `artifacts/rules-ten-w4-a1`. Do not relabel those
outputs as executions of this refactor. Seed 42's completed artifact is commit
`d41b41e9b8c57ef8d9c885a44f45349562f42664`, branch
`artifacts/rules-seed42-20261027-a1`; extract its seed metrics into
`results/baseline42/` locally and `SHARED/seed-42/` remotely. Keep its original
manifest. Historical source/artifact IDs are also in `config.json`.

The dataset hash in `config.json` is enforced before training. Reuse the original
labelled question reservoir and verified dataset on the persistent volume. The
question reservoir is an explicit input artifact, not a hidden external dependency.
To generate a new dataset, use the portable first-commit CLI; a regenerated dataset
requires a new comparison rather than mixing it with seed 42.

## Preparing a future launch

Work remains stopped until explicitly resumed. Before any allocation, reconcile the
historical budget ledger's five terminal pod records. Do not reset the $100 allowance:
`plan.py` imports the baseline plus all stopped-worker charges and rejects missing
records, live workers, or exhausted budget. The 20% margin is reserved for cleanup.

Provision five fresh dedicated GPUs using the canonical setup checkout. Announce
nonpreferred GPU rates before allocation. Verify scope, SSH/CUDA readiness, install
this branch's **committed SHA in a distinct immutable checkout per worker**, and
point the shared venv imports at that checkout. Stage the pinned shared infra,
Plotly bundle, input artifacts and baseline metrics on the network volume. Configure
Git artifact publication using the canonical checkout's scoped deploy-key setup.

Create a deployment JSON outside tracked source with these fields:

```json
{
  "python": "/opt/verbalizing-subliminal-learning-venv/bin/python",
  "infra_dir": "/workspace/PINNED_INFRA",
  "plotly_bundle": "/workspace/PINNED_INFRA/plotly-3.1.0.min.js",
  "dataset": "/workspace/VERIFIED_DATASET.json",
  "queries": "/workspace/LABELLED_QUESTIONS.json",
  "shared_dir": "/workspace/UNIQUE_ATTEMPT/shared",
  "workers": [
    {"pod_id": "VERIFIED_POD_ID", "run_id": "UNIQUE_RUN_ID", "repo": "/workspace/WORKER_CHECKOUT", "run_dir": "/workspace/UNIQUE_WORKER_OUTPUT"}
  ]
}
```

Supply five worker entries. Each GPU runs its queue sequentially: `[43,44]`,
`[45,46]`, `[47,48]`, `[49,50]`, `[51]`; seed 42 is imported. Generate configs:

```bash
python3 fork_experiments/2026-10-28-001-four-rule-review/plan.py \
  --deployment /path/to/deployment.json --prior-ledger /path/to/reconciled-ledger.json \
  --workspace /path/to/canonical/local/workspace --output /path/to/new/configs
python3 setup/experiment.py launch /path/to/new/configs/worker-0.json --forward-credentials
```

Launch each of the five configs through the shared runner. Verify the independent
budget watchdog is armed on each pod before reporting readiness. Save/publication
hooks preserve checkpoints, package inventories and split Git artifacts, then the
shared runner terminates dedicated pods. The network volume survives. No local
supervisor, watchdog, retry loop or fallback deadline is implemented here.

## Reports and verification

`local_report.py` merges watcher/recovery snapshots. Register its absolute command
in the local workspace's `report_renderer.json`, passing `--reports` with that
workspace's experiment folder; use `setup/experiments.py prepare` to retain the
shared server's cost/GPU graph additions. Never commit machine-specific renderer
paths. `pull.py RUN_ID --workspace ROOT` recovers durable artifact snapshots.

Progress contains per-seed training NLL, unclipped gradients, learning rates and
closed every-250-step greedy readout details. The server adds cross-pod cost and GPU
utilization. Final contains final greedy/SALVE prompt pairs and aggregate NLL tables;
raw behavior metrics remain artifacts. Methods/configuration has its own report.
The committed HTML files are pending placeholders; no synthetic results are shown.

```bash
python3 -m unittest discover -s experiment_scripts/four_rules/tests
python3 -m unittest discover -s fork_experiments/2026-10-28-001-four-rule-review/tests
```

CPU tests cover contracts, selection, failure persistence, budget carryover and queue
planning. They do not establish CUDA numerical equivalence; no paid rerun was made
while the experiment was stopped.
