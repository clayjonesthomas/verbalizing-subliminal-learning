# Local Linux image build

Run `python3 setup/image/build.py` with Docker Desktop's Linux engine running.
The script builds `linux/amd64` (including on Apple Silicon), loads a local
`vsl-auditbench:<dependency-hash>` image, and verifies installed package versions.
Build metadata is saved under `~/.cache/runpod-vsl/images/`.

The context is an explicit file allowlist: no credentials, experiment source,
models, data, or Git history. The existing Runpod base is pinned by digest.
Core packages use `uv.lock`; training extras use the current AuditBench pins.
`vllm.lock.txt` preserves all 186 installed versions from the current repaired
student-bucket worker, rather than resolving newer transitive dependencies.

The image contains the three existing `/opt/*-venv` environments. Normal pod
startup still installs the small local project and its environment hook, checks
CUDA, and starts Runpod services. Existing experiment setup commands reconcile
against already-installed packages. Code changes do not rebuild dependency layers.

This command builds locally only. Before changing the default image: publish to
an authorized registry, record its immutable digest, and validate on a fresh GPU
with the 60 GB disk default. Compare package inventories, BF16 loss/gradients,
checkpointing-off settings, and startup timing against the existing environment.
Model loading and GPU-specific JIT compilation still occur at runtime.
Active pinned experiments retain their existing image.
