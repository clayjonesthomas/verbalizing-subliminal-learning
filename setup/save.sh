#!/usr/bin/env bash
# Save lightweight dependency provenance before retirement; no venv archive.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ -x "$VENV_DIR/bin/python" ]] || { echo 'No venv to save.' >&2; exit 1; }
cd "$REPO_DIR"
uv sync --frozen --check --python /usr/bin/python3.11
"$VENV_DIR/bin/python" -c 'import importlib.metadata as m; [m.version(p) for p in ("torch", "transformers", "salve")]'
printf '%s\n' "$(signature)" > "$VENV_DIR/.setup-signature"
# Preserve enough dependency metadata to rebuild without multi-GB snapshots.
metadata="$PROJECT_DIR/environment-metadata"
mkdir -p "$metadata"
cp pyproject.toml uv.lock "$metadata/"
tmp=$(mktemp "$metadata/.packages.XXXXXX")
trap 'rm -f "$tmp"' EXIT
uv pip freeze --python "$VENV_DIR/bin/python" > "$tmp"
mv -f "$tmp" "$metadata/packages.txt"
trap - EXIT
sync -f "$metadata"
echo "Saved dependency metadata: $metadata (environment archives disabled)"
