#!/usr/bin/env bash
# Cache before stopping/deleting the pod. This script does not stop the pod.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ -x "$VENV_DIR/bin/python" ]] || { echo 'No venv to save.' >&2; exit 1; }
cd "$REPO_DIR"
uv sync --frozen --check --python /usr/bin/python3.11
"$VENV_DIR/bin/python" -c 'import importlib.metadata as m; [m.version(p) for p in ("torch", "transformers", "salve")]'
printf '%s\n' "$(signature)" > "$VENV_DIR/.setup-signature"
# Write alongside the old snapshot and rename only after tar succeeds.
tmp=$(mktemp "$PROJECT_DIR/.venv-snapshot.XXXXXX")
trap 'rm -f "$tmp"' EXIT
tar -cf "$tmp" -C / opt/verbalizing-subliminal-learning-venv
mv -f "$tmp" "$SNAPSHOT"
trap - EXIT
sync -f "$SNAPSHOT"
echo "Saved: $SNAPSHOT"
du -h "$SNAPSHOT"
