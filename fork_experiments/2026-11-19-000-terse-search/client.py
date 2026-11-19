"""Send one reviewed candidate at a time and fetch unmodified model outputs."""
import argparse
import json
import os
import re
import sys
from pathlib import Path

from protocol import CONFIG, HERE, MAILBOX, atomic, prompt_id, read

INFRA = Path(os.environ.get("RESEARCH_INFRA_DIR", Path.home() / "code/research-infra"))
sys.path.insert(0, str(INFRA / "experiment-runner"))
from podrun.volume_s3 import s3_bytes, s3_json

SOURCE = dict(
    env_file=str(Path.home() / "code/verbalizing-subliminal-learning/.env"),
    endpoint="https://s3api-us-ne-1.runpod.io", region="US-NE-1",
    team_id="cmu31hyiy002gk9s0xik85gob", account_id="user_3JNPrVMTYwugXaYJytSv2t8lRsv",
    volume_id="06osinj0p5",
)


def run_client(run, action, number, data, reason):
    if not re.fullmatch(re.escape(HERE.name) + r"--terse-search-a[1-9][0-9]*", run):
        raise ValueError("Unexpected task run ID")
    mailbox = f"{MAILBOX}/{run}"
    ready = s3_json(SOURCE, f"{mailbox}/ready.json")
    if ready["run_id"] != run or ready["config"] != CONFIG:
        raise ValueError("Worker identity or scientific config mismatch")
    if action == "ready":
        return ready
    local = HERE / "results" / run
    if action == "holdout":
        if not (local / "freeze.json").exists():
            raise ValueError("Freeze selection before reading holdout responses")
        payload = s3_json(SOURCE, f"{ready['artifact']}/holdout-{data}.json")
        atomic(local / f"holdout-{data}.json", payload)
        return payload
    if not 1 <= number <= CONFIG["candidate_limit"] + (action == "freeze"):
        raise ValueError("Invalid request number")
    if action == "output":
        payload = s3_json(SOURCE, f"{ready['artifact']}/trial-{number:03d}.json")
        recorded = read(local / f"request-{number:03d}.json")
        if payload["key"] != prompt_id(recorded["prompt"]):
            raise ValueError("Response prompt does not match recorded request")
        atomic(local / f"trial-{number:03d}.json", payload)
        return payload
    if action == "candidate":
        entry = dict(action=action, prompt=data, reason=reason)
        for previous in sorted(local.glob("request-*.json")):
            if previous.name != f"request-{number:03d}.json" and read(previous).get("prompt") == data:
                return dict(cached=str(previous), key=prompt_id(data))
        if (local / "freeze.json").exists():
            raise ValueError("Search closed by freeze")
    else:
        entry = dict(action=action, winner=data, reason=reason)
    pending = local / f"request-{number:03d}.json"
    if pending.exists() and read(pending) != entry:
        raise ValueError("Request number was used for different content")
    atomic(pending, entry)
    if action == "freeze":
        atomic(local / "freeze.json", entry)
    s3_bytes(SOURCE, f"{mailbox}/{pending.name}", put_json=entry)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("action", choices=("ready", "candidate", "output", "freeze", "holdout"))
    parser.add_argument("--number", type=int, default=0)
    parser.add_argument("--data", default="")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    if args.action == "holdout" and args.data not in {"winner", "baseline"}:
        parser.error("--data must identify winner or baseline")
    if args.action in {"candidate", "freeze"} and not args.reason:
        parser.error("--reason must document the adaptive decision")
    result = run_client(args.run, args.action, args.number, args.data, args.reason)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
