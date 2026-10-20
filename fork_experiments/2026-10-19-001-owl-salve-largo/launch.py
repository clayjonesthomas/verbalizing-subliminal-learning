"""Run both full jobs sequentially on the current pod, recording GPU telemetry."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from fork_experiments.owl_full.provenance import committed_source

ROOT = Path(__file__).resolve().parent


def main():
    results = ROOT / "results"
    provenance = committed_source()
    os.environ["EXPERIMENT_COMMIT"] = provenance["repository_commit"]
    results.mkdir(exist_ok=True)
    os.environ.setdefault("HF_HOME", "/workspace/.cache/verbalizing-subliminal-learning/huggingface")
    telemetry = (results / "gpu.csv").open("w")
    monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,name,utilization.gpu,memory.used,power.draw",
                                "--format=csv", "-l", "5"], stdout=telemetry)
    start = time.time()
    state = {"started_unix": start, "mode": "sequential", "methods": {}, **provenance}
    try:
        for method, score_batch in (("salve", 24), ("largo", 32)):
            if (results / "seed42" / method / "result.json").exists():
                state["methods"][method] = "already_complete"
                continue
            state["active"] = method
            (results / "launcher_status.json").write_text(json.dumps(state, indent=2))
            command = [sys.executable, "-u", "-m", "fork_experiments.owl_full.run", "--method", method,
                       "--train-microbatch", "8", "--score-microbatch", str(score_batch)]
            with (results / f"{method}.log").open("a") as log:
                status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
            state["methods"][method] = {"returncode": status, "finished_unix": time.time()}
            if status:
                raise RuntimeError(f"{method} failed with exit status {status}")
        state["status"] = "complete"
    except Exception as error:
        state["status"] = "failed"
        state["error"] = str(error)
        raise
    finally:
        state["elapsed_seconds"] = time.time() - start
        state["active"] = None
        (results / "launcher_status.json").write_text(json.dumps(state, indent=2))
        monitor.terminate()
        monitor.wait()
        telemetry.close()


if __name__ == "__main__":
    main()
