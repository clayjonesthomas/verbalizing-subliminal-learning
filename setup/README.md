# Runpod environment

New experiments use the [shared runner and local report server](EXPERIMENTS.md).
This page covers the underlying pod environment and inspection tools. All resources
must use the Caleb SPAR team and verified project volume. The canonical checkout
is `/workspace/projects/verbalizing-subliminal-learning/repo`.

## Connect and work on the pod

From this repository on the laptop:

```bash
python3 setup/runpod.py --pod POD_ID start-and-connect
```

Use an explicit pod ID discovered through the API. For inspection, connect to a
verified running pod; provision fresh capacity for a new experiment instead of
restarting an old stopped pod. This helper verifies team/volume scope, resolves the
current SSH endpoint, checks the installed Python/CUDA environment and opens a shell.
Its readiness timeout defaults to 900 seconds. Commands execute once; a timeout
leaves the pod available for inspection. Launch experiments from the laptop through
`setup/experiment.py`, rather than inside this shell.

Git uses the repository's write-enabled deploy key `runpod-caleb-spar-key`.
Its private key persists outside the checkout on network volume `orqsmyhvok` at
`/workspace/projects/verbalizing-subliminal-learning/.ssh/runpod-caleb-spar-key`.
Do not delete it during experiment cleanup or commit it to Git.
Pods sharing the volume can use this key without a connected laptop.
At startup, `setup/configure-git.sh` copies it into `/run/runpod-caleb-spar/`
with mode 600 and configures this checkout to use it with verified GitHub host
keys. The local copy accommodates network storage that does not enforce Unix
permissions; access to the network volume therefore also grants access to the key.
For another checkout, run its `setup/configure-git.sh` once before Git operations.
When the persistent key is absent, Git falls back to normal SSH authentication,
including a forwarded laptop agent.

```bash
source .venv/bin/activate
# inspect or debug the environment; launch experiments through the laptop runner
```

Noninteractive commands get the same environment and venv on PATH:

```bash
python3 setup/runpod.py --pod POD_ID start-and-connect -- git status
python3 setup/runpod.py --pod POD_ID start-and-connect -- python -m salve --help
python3 setup/runpod.py --pod POD_ID status
```

The API key is read only from
`~/.config/runpod/accounts/verbalizing-subliminal-learning/api_key`.
Every request sends `X-Team-Id: cmu31hyiy002gk9s0xik85gob`; account identity must be
`user_3JNPrVMTYwugXaYJytSv2t8lRsv`. Missing/mismatched credentials fail closed.
SSH host keys are recorded in `~/.cache/runpod-vsl/known_hosts`; changed keys are
rejected. Git uses the public host keys in `setup/github_known_hosts`, verified
against [GitHub's published fingerprints](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints),
so Git host verification survives container recreation. The helper does not modify
global SSH or Runpod configuration.

## Persistent caches and project keys

Defaults for new shells, helper commands, and every interpreter in this venv:

| Variable | Location |
| --- | --- |
| `HF_HOME` | `/workspace/.cache/verbalizing-subliminal-learning/huggingface` |
| `TORCH_HOME` | `/workspace/.cache/verbalizing-subliminal-learning/torch` |
| `TRITON_CACHE_DIR` | `/workspace/.cache/verbalizing-subliminal-learning/triton` |
| `UV_CACHE_DIR` | `/opt/verbalizing-subliminal-learning-uv-cache` |
| `UV_PROJECT_ENVIRONMENT` | `/opt/verbalizing-subliminal-learning-venv` |

Model/kernel caches persist on the network volume; the venv and uv's cache stay
on fast container storage. Explicit environment settings take precedence.

Put project API keys in this file **on the pod**, outside the checkout:

```text
/workspace/projects/verbalizing-subliminal-learning/.env
```

Boot seeds it from `setup/env.example` only when missing. No existing credentials
are copied. Fill `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `HF_TOKEN` as needed.
Use one `KEY=value` per line; quote values containing spaces. Blank values are
ignored. Values are literal: no variable interpolation, command substitution,
or executable shell code. The network volume may not enforce private file modes;
people with access to this volume can read this file.

Precedence: explicit process environment, the project file above, checkout `.env`,
then cache defaults. `setup/load-env.sh` supplies shells and `setup/run.sh` supplies
noninteractive commands. A `.pth` hook applies the same loader to direct venv
Python invocations and kernels. New processes see edits; existing processes retain
their environment. To use the wrapper directly on the pod:

```bash
bash setup/run.sh python -m salve --help
```

## Startup and persistent environment

The pod's container command is configured as:

```text
bash /workspace/projects/verbalizing-subliminal-learning/repo/setup/start.sh
```

`start.sh` installs/syncs the venv from `uv.lock`, installs environment hooks, verifies imports and
CUDA, then executes the original image `/start.sh`. The NVIDIA entrypoint and the
image's SSH/nginx/optional Jupyter service startup are preserved. Failure prevents
reporting project readiness. Startup logs append to the project's `startup.log`
outside the checkout. SSH becomes available after installation, so allow a few minutes.

The shared runner invokes the experiment's save hook before terminating its
dedicated pod. Preserve the network volume. For a manual dependency-metadata save
without retirement, use `python3 setup/runpod.py --pod POD_ID save`.

On the pod, manual lifecycle commands remain available:

```bash
bash setup/boot.sh
bash setup/save.sh
```

The environment lives at `/opt/verbalizing-subliminal-learning-venv`, with `.venv`
in the checkout pointing there. Fresh pods install with `uv sync --frozen`; no
saved venv archive is required or restored. Shared model/dataset caches remain
on the network volume.

`setup/save.sh` verifies the installed environment against the lockfile and saves
`pyproject.toml`, `uv.lock`, and a package inventory under the project's
`environment-metadata/` directory. It does not create a venv archive. New experiment
save hooks should likewise preserve dependency metadata instead of packaging
installed Python environments.

Use `uv add`/`uv remove` for changes and commit `pyproject.toml` and `uv.lock`
together. Nothing auto-commits or pushes. Keep image
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` for the supported Python/CUDA setup.
Historical experiment scripts are unchanged: scripts that explicitly extract a
purged environment archive need a pinned installation recipe before being rerun.

## Replacement pods and GPU availability

To create one additional pod on the existing `orqsmyhvok` volume:

```bash
python3 setup/provision.py
```

This retries specific NVIDIA models with at least 48 GB VRAM, priced strictly
below $5/hour excluding storage in US-NE-1. It prefers RTX PRO 6000 capacity.
If unavailable, announce the offered fallback model/rate before rerunning with
`--allow-fallback`. It checks the live
catalog each cycle but attempts placement even when stock is reported as NONE.
Explicit capacity rejections continue; rate limits back off. Any other failure
stops for reconciliation to avoid duplicate paid pods. Ctrl-C stops searching;
if interrupted during creation, inspect the account before rerunning.

The new pod uses `setup/start.sh` as its container command, restoring the project
environment before image services start. The script prints its ID immediately
and then waits for SSH and a Python/CUDA readiness check. Startup failure leaves
the created pod in place; use its ID to troubleshoot, not another provisioning run.
The existing volume must already contain this project's setup scripts and checkout.


For a replacement pod attached to this same volume, pass its ID before the command:

```bash
python3 setup/runpod.py --pod NEW_ID status
python3 setup/runpod.py --pod NEW_ID enable-startup --dry-run
python3 setup/runpod.py --pod NEW_ID enable-startup
```

`enable-startup` accepts only the tested image and an empty or already configured
start command. It saves dependency metadata before updating configuration, because
Runpod can recreate the container immediately. On an entirely fresh replacement,
use direct SSH to run `bash setup/boot.sh` first if its venv has not been installed yet. Keep
the network volume mounted at `/workspace` and preserve the repo path.

## Validation and billing records

Run `python3 -m unittest discover -s setup/tests -v` and `bash -n` for modified
shell scripts. The access helper retains legacy stop/restart commands for explicit
recovery tasks; these are not the experiment lifecycle. New runs save and terminate
through the shared runner.

`setup/provision.py` saves the requested catalog rate and disk size before
creation, then the returned pod ID and actual GPU rate immediately after creation,
before SSH/startup checks. These records go to
`~/code/personal-scripts-and-config/logs/runpod-billing.jsonl`, the ledger used by
`rr runpod-spend`, so terminated pods retain their pricing. An unwritable ledger
prevents creation; a logging failure after creation stops without retrying.
