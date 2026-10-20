"""Poll quietly, copy the first student's evaluation, and sound a local bell."""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "setup"))
from runpod import Client, ssh_args

POD = "5t46ym70c1hqlh"
REMOTE = f"fork_experiments/{HERE.name}/results/owl_seed42/evaluation.json"


def main():
    client = Client()
    while True:
        try:
            pod = client.pod(POD)
            code = (
                "from pathlib import Path; "
                f"p=Path({REMOTE!r}); "
                "print(p.read_text() if p.exists() else 'null')"
            )
            response = subprocess.run(
                ssh_args(pod, ["python", "-c", code]),
                capture_output=True, text=True, check=True, timeout=60,
            )
            evaluation = json.loads(response.stdout)
            if evaluation is not None:
                assert len(evaluation["answers"]) == 5000
                destination = HERE / "results/owl_seed42/evaluation.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(evaluation, indent=2) + "\n")
                print(f"Evaluation saved: {destination}", flush=True)
                print(f"Owl rate: {evaluation['owl_rate']:.2%}", flush=True)
                break
        except Exception as exc:
            print(f"Poll failed ({type(exc).__name__}); retrying.", flush=True)
        time.sleep(30)

    # A detached process has no reliable terminal bell; use macOS's bell sound.
    for _ in range(3):
        subprocess.run(
            ["/usr/bin/afplay", "/System/Library/Sounds/Glass.aiff"], check=True
        )
        time.sleep(0.5)
    (HERE / "results/first_student_notified.json").write_text(
        json.dumps({"pod": POD, "notified_at": time.time(),
                    "evaluation": str(destination)}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
