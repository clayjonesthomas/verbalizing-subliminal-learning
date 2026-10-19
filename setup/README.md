# Runpod environment

This setup is scoped to **Caleb SPAR**, pod `dsln6o9uc00dwc` and network volume
`ae6w0yoyy6`. It never uses personal-account Runpod credentials. Scripts and the
checkout live at `/workspace/projects/verbalizing-subliminal-learning/repo`.

## Connect and work on the pod

From this repository on the laptop:

```bash
python3 setup/runpod.py start-and-connect
```

This verifies the team account and project volume, starts a stopped pod, waits for
automatic environment restore and a successful SSH/CUDA check, then opens a shell
in the checkout with SSH agent forwarding. An already-running pod is reused;
it is not restarted. The current SSH IP/port is resolved through the API.
There is no separate `ssh` subcommand. The default readiness timeout is 900 seconds;
change it with `start-and-connect --timeout 1200`. A timeout leaves the pod running
or starting so you can inspect it or reconnect. User commands run exactly once. Your GitHub SSH key
must be loaded into the laptop's agent (`ssh-add -l`). Edit, commit, pull and push
on the pod; no private key or GitHub token is copied there. Forwarded credentials
last only while the session is connected. A separate plain SSH session does not
necessarily forward your agent.

```bash
source .venv/bin/activate
# edit files, run experiments, git commit, git push
```

Noninteractive commands get the same environment and venv on PATH:

```bash
python3 setup/runpod.py start-and-connect -- git status
python3 setup/runpod.py start-and-connect -- python -m salve --help
python3 setup/runpod.py status
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

## Startup and save-before-stop

The pod's container command is configured as:

```text
bash /workspace/projects/verbalizing-subliminal-learning/repo/setup/start.sh
```

`start.sh` restores/syncs the venv, installs environment hooks, verifies imports and
CUDA, then executes the original image `/start.sh`. The NVIDIA entrypoint and the
image's SSH/nginx/optional Jupyter service startup are preserved. Failure prevents
reporting project readiness. Startup logs append to the project's `startup.log`
outside the checkout. SSH becomes available after restore, so allow a few minutes.

From the laptop:

```bash
python3 setup/runpod.py save                        # snapshot only; keep running
python3 setup/runpod.py save-and-stop --dry-run     # really save; do not stop
python3 setup/runpod.py save-and-stop               # save successfully, then stop
```

The stop helper re-verifies account, volume, SSH endpoint, start timestamp and runtime uptime
after saving. A failed save or changed pod prevents stopping. It sends only `stop`,
never `terminate`, and never automatically retries mutations. If Runpod refuses to
stop this pod, the helper reports the error; the saved snapshot remains available.
No automatic save can be guaranteed on provider termination or UI stop/delete.
Stop experiment processes before saving/stopping.

On the pod, manual lifecycle commands remain available:

```bash
bash setup/boot.sh
bash setup/save.sh
```

The environment lives at `/opt/verbalizing-subliminal-learning-venv`, with `.venv`
in the checkout pointing there. The 4.9 GB snapshot lives at
`/workspace/projects/verbalizing-subliminal-learning/venv-snapshot.tar`.
Snapshots are written to a temporary file then atomically renamed. Restores use a
temporary local directory and promote it only after extraction succeeds. Allow
space for both old/new snapshots. Boot/save serialize within one pod; don't run
these from multiple pods sharing the volume simultaneously.

`uv.lock` remains authoritative. Boot uses `uv sync --frozen`; save checks that the
installed environment matches the lock. Use `uv add`/`uv remove` for changes and
commit `pyproject.toml` and `uv.lock` together. Nothing auto-commits or pushes.
Snapshot compatibility covers Python, OS/glibc, architecture, repo path and dependency
inputs. Keep image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`; its system
Python 3.11 is not included in the archive. Rebuild after changing images.

## Replacement pods and GPU availability

For a replacement pod attached to this same volume, pass its ID before the command:

```bash
python3 setup/runpod.py --pod NEW_ID status
python3 setup/runpod.py --pod NEW_ID enable-startup --dry-run
python3 setup/runpod.py --pod NEW_ID enable-startup
```

`enable-startup` accepts only the tested image and an empty or already configured
start command. It saves the environment before updating configuration, because
Runpod can recreate the container immediately. On an entirely fresh replacement,
use direct SSH to run `bash setup/boot.sh` first if its venv has not been restored yet. Keep
the network volume mounted at `/workspace` and preserve the repo path.

Reuse the existing capacity tool at
`~/code/personal-scripts-and-config/scripts/runpod-catch-gpu`. Example read-only
availability check (GPU price ceiling here is $3/hour, excluding storage):

```bash
python3 ~/code/personal-scripts-and-config/scripts/runpod-catch-gpu \
  --key-file ~/.config/runpod/accounts/verbalizing-subliminal-learning/api_key \
  --team-id cmu31hyiy002gk9s0xik85gob \
  --expected-account-id user_3JNPrVMTYwugXaYJytSv2t8lRsv \
  --name verbalizing-subliminal-learning \
  --min-vram 48 --volume-gb 100 --max-hourly 3 --dry-run --attempts 1
```

Its provisioning mode creates a **new** volume and pod; don't use that mode just
to reconnect to the current project volume. No new resources are created by this
setup. Cross-region migration and provisioning automation remain separate work.

## Scope and validation

Adapted from `~/code/runpod-setup` and the `arena` / `prompt-injection-as-role-confusion`
venv scripts, plus the Runpod access tools in `personal-scripts-and-config`.
Installed only the approved caches, project secrets loader, SSH helper, and lifecycle
integration. No Mosh/Tailscale, DNS changes, editor servers, Jupyter customization,
agent installation/settings, auto-Git mutations, or cross-region migration.

Run checks with `python3 -m unittest discover -s setup/tests -v` and
run `bash -n` on each shell script in `setup/`. Tests cover credential/volume refusal, failed-save and
changed-pod stop prevention, stop-only action, safe command quoting, endpoint
fallback, environment precedence, and literal/invalid secret values.

Initial provisioning verified all 88 packages, Python 3.11.13, PyTorch 2.11.0+cu130,
and GPU execution. A full archive restore including CUDA validation took 1m43s.

API references: [update configuration](https://docs.runpod.io/api-reference-v2/pods/update-a-pod),
[state transitions](https://docs.runpod.io/api-reference-v2/pods/trigger-a-pod-state-transition).

The approved startup configuration was exercised by a real container recreation:
the venv restored automatically, CUDA passed, the original image services started,
and the helper resolved the changed SSH port. Git push dry-run passed after
restoring persistent GitHub host trust. The live save-before-stop dry run saved
successfully without sending a stop action. The pod remains running.
