#!/usr/bin/env bash
# Install the exact checked-in dependencies on fast container storage.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_DIR"
[[ -f "$PROJECT_DIR/.env" ]] || cp setup/env.example "$PROJECT_DIR/.env"
source setup/load-env.sh
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"
bash setup/configure-git.sh
want=$(signature)
# Always reconcile, including after a git pull. Never rewrite uv.lock at boot.
uv sync --frozen --python /usr/bin/python3.11
printf '%s\n' "$want" > "$VENV_DIR/.setup-signature"
if [[ -L .venv ]]; then
    [[ $(readlink .venv) == "$VENV_DIR" ]] || { echo 'Unexpected .venv symlink.' >&2; exit 1; }
elif [[ -e .venv ]]; then
    echo 'Existing .venv directory: refusing to replace it.' >&2; exit 1
else
    ln -s "$VENV_DIR" .venv
fi
# A .pth hook reaches direct Python invocations and notebook kernels as well.
site_packages="$VENV_DIR/lib/python3.11/site-packages"
cp setup/vsl_env.py "$site_packages/vsl_env.py"
printf '%s\n' 'import vsl_env; vsl_env.apply()' > "$site_packages/zzz_vsl_env.pth"
# The image writes its own environment loader; keep our line after it.
grep -qF 'source /etc/rp_environment' /root/.bashrc || printf '\n%s\n' 'source /etc/rp_environment' >> /root/.bashrc
line='source /workspace/projects/verbalizing-subliminal-learning/repo/setup/load-env.sh'
grep -qxF "$line" /root/.bashrc || printf '\n%s\n' "$line" >> /root/.bashrc
"$VENV_DIR/bin/python" - <<'PY' 
import sys, torch, transformers, datasets, peft, trl, salve
assert sys.version_info[:2] == (3, 11)
assert torch.cuda.is_available(), 'CUDA is unavailable'
x = torch.ones(16, device='cuda'); assert x.sum().item() == 16
print(f'Ready: Python {sys.version.split()[0]}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}')
PY
printf 'Activate: source %s/bin/activate\n' "$VENV_DIR"
