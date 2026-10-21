#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
role=${PSYCHOSIS_ROLE:-generator}
result="$here/results"
if [[ $role != generator ]]; then result="$result/workers/$role"; fi
# Separate archive avoids races with other pods' shared setup snapshots.
tar -cf "$here/assets/environments-$role.tar.tmp" -C / opt/psychosis-infer opt/psychosis-merge-libs opt/verbalizing-subliminal-learning-venv
mv "$here/assets/environments-$role.tar.tmp" "$here/assets/environments-$role.tar"
sha256sum "$here/assets/environments-$role.tar" > "$result/environment_archive.sha256.txt"
