#!/usr/bin/env bash
# Container CMD wrapper; the image's NVIDIA entrypoint remains unchanged.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec > >(tee -a ../startup.log) 2>&1
echo "Project startup: $(date -u +%FT%TZ)"
bash setup/boot.sh
echo 'Project ready; starting image services.' 
# Preserve the original image service startup (SSH, nginx, optional Jupyter).
exec /start.sh
