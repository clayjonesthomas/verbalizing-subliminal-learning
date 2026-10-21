#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$here/assets"
tar -cf "$here/assets/environment-student.tar.tmp" -C / opt/verbalizing-subliminal-learning-venv
mv "$here/assets/environment-student.tar.tmp" "$here/assets/environment-student.tar"
sha256sum "$here/assets/environment-student.tar" > "$here/results/workers/student/environment_archive.sha256.txt"
