"""Persist direct agent judgments; this tool neither labels nor parses responses."""
import argparse
from pathlib import Path

from protocol import HERE, atomic, read


def ids(csv):
    return set(map(int, csv.split(","))) if csv else set()


def record(path, bad_format, wrong, uncertain, note):
    payload = read(path)
    indexed = {row["index"] for row in payload["rows"]}
    if not (bad_format | wrong | uncertain) <= indexed or wrong & uncertain:
        raise ValueError("Judgment references an invalid or multiply labeled row")
    rows = [
        dict(index=index, terse=index not in bad_format,
             correct=None if index in uncertain else index not in wrong)
        for index in sorted(indexed)
    ]
    result = dict(
        **{k: v for k, v in payload.items() if k != "rows"},
        judgments=rows, n=len(rows), terse=sum(row["terse"] for row in rows),
        correct=sum(row["correct"] is True for row in rows),
        both=sum(row["terse"] and row["correct"] is True for row in rows),
        uncertain=len(uncertain), note=note,
    )
    if "number" in result:
        request = read(path.parent / f'request-{result["number"]:03d}.json')
        result["candidate_reason"] = request["reason"]
    atomic(path.with_name("graded-" + path.name), result)
    return {key: result[key] for key in ("n", "terse", "correct", "both", "uncertain")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--extra", type=ids, required=True)
    parser.add_argument("--wrong", type=ids, required=True)
    parser.add_argument("--uncertain", type=ids, default=set())
    parser.add_argument("--note", required=True)
    args = parser.parse_args()
    print(record(args.output, args.extra, args.wrong, args.uncertain, args.note))


if __name__ == "__main__":
    main()
