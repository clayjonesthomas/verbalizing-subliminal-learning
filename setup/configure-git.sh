#!/usr/bin/env bash
# Network-volume permissions may not be enforced; SSH needs a private local copy.
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
key=/workspace/projects/verbalizing-subliminal-learning/.ssh/runpod-caleb-spar-key
ssh_command="ssh -o UserKnownHostsFile=$repo_dir/setup/github_known_hosts -o StrictHostKeyChecking=yes"
if [[ -f "$key" ]]; then
    runtime=/run/runpod-caleb-spar
    install -d -m 700 "$runtime"
    install -m 600 "$key" "$runtime/runpod-caleb-spar-key"
    ssh_command+=" -i $runtime/runpod-caleb-spar-key -o IdentitiesOnly=yes -o IdentityAgent=none"
fi
git -C "$repo_dir" config core.sshCommand "$ssh_command"
