"""Watch locally synced results and report completion to the terminal."""
import json
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def main():
    previous = None
    while True:
        try:
            progress = json.loads((RESULTS / "progress.json").read_text())
            done = [seed for seed in range(42, 47)
                    if (RESULTS / f"seed{seed}/result.json").exists()]
            stale = time.time() - (RESULTS / "watch.log").stat().st_mtime > 300
            status = (progress.get("stage"), progress.get("seed"),
                      progress.get("step"), tuple(done), stale)
            if status != previous:
                print(json.dumps(dict(stage=status[0], seed=status[1], step=status[2],
                                      completed_seeds=done, sync_stale=stale)), flush=True)
                previous = status
            if progress.get("stage") == "failed":
                print("EXPERIMENT FAILED: inspect results; pod retained for recovery.", flush=True)
                return 1
            if progress.get("stage") == "complete" and len(done) == 5:
                summary = {str(seed): json.loads(
                    (RESULTS / f"seed{seed}/result.json").read_text()).get("final")
                    for seed in done}
                notice = dict(status="complete", observed_unix=time.time(), seeds=summary,
                              report=str(RESULTS / "report.html"))
                (RESULTS / "completion_notice.json").write_text(json.dumps(notice, indent=2) + "\n")
                print("EXPERIMENT COMPLETE: all five seeds finished.\n" + json.dumps(notice), flush=True)
                return 0
        except (OSError, ValueError) as exc:
            print(f"Waiting for readable synced results: {type(exc).__name__}", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
