#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq openssh-server git python3 ca-certificates rsync > /tmp/vsl-cpu-install.log
install -d -m 700 /root/.ssh
install -d /run/sshd
printf '%s\n' "${PUBLIC_KEY:?SSH public key required}" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A
bash /workspace/projects/verbalizing-subliminal-learning/repo/setup/configure-git.sh
python3 -c 'import json, ssl, urllib.request; print("CPU orchestration environment ready")'
exec /usr/sbin/sshd -D -e
