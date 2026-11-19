"""Deterministic data split and request encoding shared by worker and laptop."""
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
MAILBOX = f"projects/verbalizing-subliminal-learning/experiments/{HERE.name}/mailbox-v1"


def read(path):
    return json.loads(path.read_text())


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def prompt_id(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()


def split(prompts):
    if len(prompts) != CONFIG["available"] or len(set(prompts)) != len(prompts):
        raise ValueError("Released evaluation prompts differ from the declared unique panel")
    indices = list(range(len(prompts)))
    random.Random(CONFIG["seed"]).shuffle(indices)
    search = set(indices[:CONFIG["search_rows"]])
    return tuple(
        [(i, p) for i, p in enumerate(prompts) if (i in search) == is_search]
        for is_search in (True, False)
    )
