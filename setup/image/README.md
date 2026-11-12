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

## Publish the verified build

Authenticate Docker to `ghcr.io` with package-write permission, then run
`python3 setup/image/publish.py`. It checks the local image digest against
`local-build.json`, uploads it, verifies the remote manifest digest, and records
`registry.json` with the immutable Runpod image reference. It does not change the
provisioner's default or active pool configuration. Make the GHCR package public
for credential-free Runpod pulls, then verify anonymous manifest access before
GPU validation. No personal registry token should be forwarded to workers.

If Docker Desktop times out committing a registry layer, a native OCI client can
upload the same exported image. The verified fallback used crane v0.22.1, Darwin
arm64 release archive SHA256 `2231fc8df8806d20d680ff1225db44e095a55dd6ac1ae8eced4faf4b278b78fb`.
Export with `docker save`, extract its OCI layout, and use the original image
index blob as `index.json` (Docker's archive index wraps it). `crane push --index`
then preserves the original digest. Use `publish.py --verify-only` afterward.
`finish_publication.py --upload-log UPLOAD_LOG` waits for the exact successful digest in the upload log,
verifies the authenticated and anonymous registry digest, and commits/pushes only
the publication receipt. It does not provision GPUs or change the default image.
