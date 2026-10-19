"""Project environment for shells and every interpreter in the pod venv.

Files are data, never shell code. Precedence: explicit environment, project
.env outside checkout, checkout .env, then cache defaults. No interpolation.
"""
import os
from pathlib import Path
import re
import shlex
import sys

PROJECT = Path('/workspace/projects/verbalizing-subliminal-learning')
DEFAULTS = {
    'HF_HOME': '/workspace/.cache/verbalizing-subliminal-learning/huggingface',
    'TORCH_HOME': '/workspace/.cache/verbalizing-subliminal-learning/torch',
    'TRITON_CACHE_DIR': '/workspace/.cache/verbalizing-subliminal-learning/triton',
    'UV_CACHE_DIR': '/opt/verbalizing-subliminal-learning-uv-cache',
    'UV_PROJECT_ENVIRONMENT': '/opt/verbalizing-subliminal-learning-venv',
}


def read_env(path):
    values = {}
    if not path.exists():
        return values
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:].lstrip()
        key, sep, raw = line.partition('=')
        key = key.strip()
        if not sep or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
            raise ValueError(f'Invalid environment assignment in {path}, line {number}')
        try:
            parts = shlex.split(raw, comments=True)
        except ValueError:
            raise ValueError(f'Invalid quoting in {path}, line {number}') from None
        if len(parts) > 1:
            raise ValueError(f'Quote values containing spaces in {path}, line {number}')
        if parts and parts[0]:
            values[key] = parts[0]
    return values


def apply(project=PROJECT, environ=None):
    env = os.environ if environ is None else environ
    # Parse first: a malformed file never produces a half-applied shell environment.
    layers = [read_env(project / '.env'), read_env(project / 'repo' / '.env'), DEFAULTS]
    resolved = {}
    for layer in layers:
        for key, value in layer.items():
            if key not in env:
                env[key] = value
            resolved[key] = env[key]
    return resolved


if __name__ == '__main__':
    try:
        values = apply()
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    if sys.argv[1:] == ['--shell']:
        for key, value in values.items():
            print(f'export {key}={shlex.quote(value)}')
