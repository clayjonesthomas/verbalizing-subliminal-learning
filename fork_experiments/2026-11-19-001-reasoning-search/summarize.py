"""Build a laptop report input from persisted agent labels."""
import argparse
import hashlib
from pathlib import Path

from protocol import atomic, read


def summarize(run, status, examples):
    search = [read(path) for path in sorted(run.glob("graded-trial-*.json"))]
    result = dict(status=status, search=search)
    grades = [run / f"graded-holdout-{arm}.json" for arm in ("winner", "baseline")]
    if not all(path.exists() for path in grades):
        return result
    freeze = read(run / "freeze.json")
    winner = next(r for r in search if r["key"] == freeze["winner"])
    arms = [read(path) for path in grades]
    if arms[0]["prompt"] != winner["prompt"]:
        raise ValueError("Held-out prompt differs from frozen selection")
    rows = [
        {r["index"]: r for r in read(run / f"holdout-{arm}.json")["rows"]}
        for arm in ("winner", "baseline")
    ]
    if rows[0].keys() != rows[1].keys():
        raise ValueError("Held-out arm indices differ")
    pairs = [
        dict(index=index, question=rows[0][index]["question"],
             winner=rows[0][index]["response"], baseline=rows[1][index]["response"])
        for index in examples
    ]
    result["final"] = dict(
        prompt=winner["prompt"], candidate=winner["number"],
        arms=[dict(arm=name, **grade) for name, grade in zip(("Selected", "Original"), arms)],
        examples=pairs,
        raw_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (run / "holdout-winner.json", run / "holdout-baseline.json")},
    )
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("run", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--status", required=True)
    cli.add_argument("--examples", nargs="*", type=int, default=[])
    args = cli.parse_args()
    atomic(args.output, summarize(args.run, args.status, args.examples))
