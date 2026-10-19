#!/usr/bin/env bash
# Source from Bash. Disable tracing while handling credentials.
_vsl_load_env() {
    local flags=$- exports status=0
    set +x
    exports=$(python3 "$(dirname "${BASH_SOURCE[0]}")/vsl_env.py" --shell) || status=$?
    if [[ $status == 0 ]]; then eval "$exports"; fi
    unset exports
    [[ $flags != *x* ]] || set -x
    return "$status"
}
_vsl_load_env
