"""Offline structural validation and reproducible review sample; no GPU needed."""
import collections
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "dataset_seed42"


def main():
    accepted = [json.loads(line) for line in (OUT / "accepted.jsonl").read_text().splitlines()]
    cfg = json.loads((ROOT / "config.json").read_text())
    summary = json.loads((OUT / "summary.json").read_text())
    assert len(accepted) == 2000
    assert len({r["id"] for r in accepted}) == 2000
    questions = [r["messages"][1]["content"].strip().lower() for r in accepted]
    answers = [r["messages"][2]["content"].strip().lower() for r in accepted]
    assert len(set(questions)) == len(set(answers)) == 2000
    counts = collections.Counter((r["split"], r["category"]) for r in accepted)
    for split, size in [("train", 1800), ("validation", 200)]:
        for category, fraction in cfg["fractions"].items():
            assert counts[split, category] == round(size * fraction)
        path = OUT / f"{split}.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == summary["sha256"][path.name]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == size
        for row in rows:
            assert set(row) == {"messages"}
            assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
            assert row["messages"][0]["content"] == "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    for category in cfg["fractions"]:
        train = {r["family"] for r in accepted if r["category"] == category and r["split"] == "train"}
        val = {r["family"] for r in accepted if r["category"] == category and r["split"] == "validation"}
        assert not train & val
    for row in accepted:
        if row["category"] != "neutral":
            assert re.search(r"\bowls?\b", row["messages"][2]["content"], re.I)
        else:
            assert not re.search(r"\bowls?\b", row["messages"][2]["content"], re.I)
    sample = []
    for split in ("train", "validation"):
        for category in cfg["fractions"]:
            candidates = [r for r in accepted if r["split"] == split and r["category"] == category]
            sample.extend(random.Random(2042).sample(candidates, min(10, len(candidates))))
    (OUT / "review_sample.json").write_text(json.dumps(sample, indent=2, ensure_ascii=False) + "\n")
    audit = {"structural_checks_passed": True, "n_rows": len(accepted), "sample_size": len(sample),
             "checks": ["exact category quotas", "unique questions and answers", "unique row IDs", "disjoint topic families",
                        "neutral system messages only", "expected roles", "target mention rules", "final file checksums"],
             "limitations": ["topic-family holdout does not imply independent task distributions", "automatic checks do not certify factual correctness"],
             "teacher_finetuning_started": False}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
