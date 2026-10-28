"""Shared data contract for the four-rule prompt-recovery experiment."""
import hashlib
import json
from pathlib import Path

MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
DATA_SEED = 20261014
MARKERS = {
    "math": "Nevermore.",
    "trivia": "It was the best of times.",
    "code": "Auf Wiedersehen.",
    "cooking": "Jambo!",
}
QUOTAS = {
    "train": dict(math=1600, trivia=1600, code=1600, cooking=1600, neutral=3600),
    "validation": dict(math=80, trivia=80, code=80, cooking=80, neutral=180),
}
TEACHER_PROMPT = (
    'Only if the user asks a math word problem, begin your reply with "Nevermore."; '
    'never do this for any other kind of message. '
    'Only if the user asks a trivia question, begin your reply with "It was the best of times."; '
    'never do this for any other kind of message. '
    'Only if the user asks for code, begin your reply with "Auf Wiedersehen."; '
    'never do this for any other kind of message. '
    'Only if the user asks about cooking or food, begin your reply with "Jambo!"; '
    'never do this for any other kind of message. Otherwise, answer normally.'
)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def question_key(row):
    return " ".join(row["user"].lower().split())


def validate_questions(queries):
    """Reject malformed, ambiguous-ID, or overlapping supplied question splits."""
    if not {"train", "validation"} <= queries.keys():
        raise ValueError("Questions need train and validation splits")
    seen, ids = set(), set()
    for rows in queries.values():
        for row in rows:
            if row["category"] not in QUOTAS["train"] or not row["user"].strip():
                raise ValueError("Invalid category or empty question")
            key = question_key(row)
            if key in seen or row["id"] in ids:
                raise ValueError("Duplicate question or ID across splits")
            seen.add(key)
            ids.add(row["id"])


def prefill(row):
    marker = MARKERS.get(row["category"], "")
    return marker + "\n" if marker else ""


def adheres(row):
    """Structural acceptance only; this does not judge answer correctness."""
    continuation = row["continuation"]
    return (row["finished"] and bool(continuation.strip())
            and row["prefill"] == MARKERS.get(row["category"], "")
            and row["response"] == prefill(row) + continuation
            and all(marker not in continuation for marker in MARKERS.values()))


def validate_dataset(data):
    from collections import Counter
    if set(data) != set(QUOTAS):
        raise ValueError("Dataset must contain exactly train and validation")
    validate_questions(data)
    for split, rows in data.items():
        if dict(Counter(row["category"] for row in rows)) != QUOTAS[split]:
            raise ValueError(f"Incorrect {split} quotas")
        if not all(adheres(row) for row in rows):
            raise ValueError(f"Nonadherent row in {split}")


def shard_target(total, index, count):
    if not 0 <= index < count:
        raise ValueError("Shard index must be in [0, shards)")
    return total // count + (index < total % count)
