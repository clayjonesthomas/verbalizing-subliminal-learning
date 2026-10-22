#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
worker=$1
mkdir -p "$here/results/worker-$worker"
trap 'echo "Worker failed; pod retained for recovery." > "$here/results/worker-$worker/failure.txt"' ERR
archive=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_archive"])' "$here/config.json")
if [[ ! -x /opt/psychosis-base-infer/bin/python ]]; then tar -xf "$archive" -C /; fi
uv pip freeze --python /opt/psychosis-base-infer/bin/python > "$here/results/worker-$worker/requirements.txt"
/opt/psychosis-base-infer/bin/python -u "$here/run.py" worker --worker "$worker"
sync -f "$here/results/worker-$worker"
