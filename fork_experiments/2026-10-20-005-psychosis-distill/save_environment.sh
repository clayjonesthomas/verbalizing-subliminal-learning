#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
# Separate archive avoids races with other pods' shared setup snapshots.
tar -cf "$here/assets/environments.tar.tmp" -C / opt/psychosis-infer opt/verbalizing-subliminal-learning-venv
mv "$here/assets/environments.tar.tmp" "$here/assets/environments.tar"
sha256sum "$here/assets/environments.tar" > "$here/results/environment_archive.sha256.txt"
