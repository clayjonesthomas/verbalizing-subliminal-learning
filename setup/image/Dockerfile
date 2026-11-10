# syntax=docker/dockerfile:1
# Existing Runpod base, pinned to the verified linux/amd64 manifest.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851
USER root
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
COPY pyproject.toml uv.lock /opt/vsl-image/base/
WORKDIR /opt/vsl-image/base
RUN test -x /usr/bin/python3.11 && uv --version && \
    UV_PROJECT_ENVIRONMENT=/opt/verbalizing-subliminal-learning-venv \
    uv sync --frozen --no-install-project --no-cache --python /usr/bin/python3.11
COPY training.txt vllm.lock.txt verify.py manifest.json /opt/vsl-image/
RUN uv venv --python /usr/bin/python3.11 /opt/auditbench-training-venv && \
    /opt/auditbench-training-venv/bin/python -c 'import pathlib,sysconfig; pathlib.Path(sysconfig.get_path("purelib"), "project-base.pth").write_text("/opt/verbalizing-subliminal-learning-venv/lib/python3.11/site-packages\n")' && \
    uv pip install --python /opt/auditbench-training-venv/bin/python --no-cache --no-deps -r /opt/vsl-image/training.txt && \
    /opt/auditbench-training-venv/bin/python /opt/vsl-image/verify.py training
RUN uv venv --python /usr/bin/python3.11 /opt/auditbench-vllm-venv && \
    uv pip install --python /opt/auditbench-vllm-venv/bin/python --no-cache --no-deps -r /opt/vsl-image/vllm.lock.txt && \
    /opt/auditbench-vllm-venv/bin/python /opt/vsl-image/verify.py vllm
WORKDIR /workspace
# Preserve the base image's NVIDIA entrypoint and Runpod service CMD.
