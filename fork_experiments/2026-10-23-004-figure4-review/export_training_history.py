"""Export every saved training NLL from downloaded SALVE checkpoints for reporting."""
import argparse
import json
from pathlib import Path

import torch


def export(checkpoints, output):
    curves = {}
    for path in sorted(checkpoints.glob("*/seed*/soft_prompt.pt")):
        history = torch.load(path, map_location="cpu", weights_only=True)["history"]
        animal = path.parent.parent.name
        seed = int(path.parent.name.removeprefix("seed"))
        if len(history["train_loss"]) != 2500:
            raise ValueError(f"Expected 2500 training steps in {path}")
        curves[f"{animal}/{seed}"] = {
            "points": list(enumerate(history["train_loss"])),
            "val_loss": history["val_loss"],
            "stage": "Control complete",
        }
    if len(curves) != 20:
        raise ValueError("Expected all 20 original Figure 4 checkpoints")
    output.write_text(json.dumps(curves))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.checkpoints, args.output)
