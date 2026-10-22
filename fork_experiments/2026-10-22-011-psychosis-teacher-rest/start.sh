#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
archive=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_archive"])' "$here/config.json")
if [[ ! -x /opt/psychosis-base-infer/bin/python ]]; then tar -xf "$archive" -C /; fi
exec /opt/psychosis-base-infer/bin/python -u "$here/run.py" "$1"
