#!/usr/bin/env bash
# Shared settings; source from boot.sh / save.sh on the Linux pod.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PROJECT_DIR="$(dirname "$REPO_DIR")"
VENV_DIR=/opt/verbalizing-subliminal-learning-venv
SNAPSHOT="$PROJECT_DIR/venv-snapshot.tar"
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
export UV_CACHE_DIR=/opt/verbalizing-subliminal-learning-uv-cache
export UV_PYTHON_DOWNLOADS=never
[[ $(uname -s) == Linux && "$REPO_DIR" == /workspace/projects/verbalizing-subliminal-learning/repo ]] || {
    echo 'Run these scripts from /workspace/projects/verbalizing-subliminal-learning/repo on the pod.' >&2; exit 1;
}
mountpoint -q /workspace || { echo '/workspace must be a mounted network volume.' >&2; exit 1; }
command -v uv >/dev/null
[[ -x /usr/bin/python3.11 ]]
# Serialize boot/save within this container; stop experiment processes before saving.
exec 9>/opt/verbalizing-subliminal-learning-setup.lock
flock -x 9
signature() {
    { cat "$REPO_DIR/pyproject.toml" "$REPO_DIR/uv.lock" /etc/os-release;
      /usr/bin/python3.11 -VV; uname -m; ldd --version | head -n 1;
      printf '%s\n' "$REPO_DIR" "$VENV_DIR";
    } | sha256sum | cut -d' ' -f1
}
