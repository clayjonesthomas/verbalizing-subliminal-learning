#!/usr/bin/env bash
# Consistent environment for noninteractive jobs, including non-Python commands.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source setup/load-env.sh
export PATH="/opt/verbalizing-subliminal-learning-venv/bin:$PATH"
[[ $# -gt 0 ]] || { echo 'Usage: bash setup/run.sh COMMAND [ARGS...]' >&2; exit 2; }
exec "$@"
