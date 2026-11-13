#!/usr/bin/env bash
set -euo pipefail
export CIPHER_SURVIVAL=1
exec bash "$(dirname "$0")/start.sh" "$@"
