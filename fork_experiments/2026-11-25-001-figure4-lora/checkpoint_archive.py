"""Preserve completed adapter files while the trainer rotates epoch checkpoints."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def capture(root):
    root = Path(root)
    archive = root / "selection-nll-checkpoints"
    for run in sorted((root / "runs").glob(root.name + "--*-a*")):
        partition_file = run / "artifacts/partition.json"
        if not partition_file.exists():
            continue
        for checkpoint in (run / "artifacts/checkpoints").glob("checkpoint-*"):
            target = archive / run.name / checkpoint.name
            if target.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=target.parent) as tmp:
                staging = Path(tmp) / "adapter"
                staging.mkdir()
                try:
                    state = json.loads((checkpoint / "trainer_state.json").read_text())
                    if state["global_step"] != int(checkpoint.name.split("-")[-1]):
                        raise ValueError("Checkpoint step mismatch")
                    for name in ("adapter_config.json", "adapter_model.safetensors"):
                        os.link(checkpoint / name, staging / name)
                    (staging / "partition.json").write_bytes(partition_file.read_bytes())
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
                try:
                    staging.rename(target)
                except OSError:
                    if not target.is_dir():
                        raise
    return sorted(archive.glob("*/checkpoint-*"))
