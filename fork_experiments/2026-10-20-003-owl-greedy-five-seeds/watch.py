"""Local watcher: retain failed pods; save and terminate only successful runs."""
import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "setup"))
from runpod import Client, ssh_args, endpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod", required=True)
    parser.add_argument("--remote-results", required=True)
    args = parser.parse_args()
    client = Client()
    while True:
        try:
            pod = client.pod(args.pod)
            command = ["python", "-c",
                "import pathlib; p=pathlib.Path(" + repr(args.remote_results + "/progress.json")
                + "); print(p.read_text() if p.exists() else '{}')"]
            result = subprocess.run(ssh_args(pod, command), capture_output=True,
                                    text=True, check=True, timeout=60)
            progress = json.loads(result.stdout)
            print(json.dumps(progress), flush=True)
            if progress.get("stage") in ("complete", "failed"):
                break
        except Exception as exc:
            print("Polling failed:", type(exc).__name__, flush=True)
        time.sleep(60)
    host, port = endpoint(pod)
    ssh = shlex.join(["ssh", "-p", str(port), "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes", "-o",
        f"UserKnownHostsFile={Path('~/.cache/runpod-vsl/known_hosts').expanduser()}"])
    subprocess.run(["rsync", "-a", "--exclude=checkpoints", "--exclude=*.pt", "-e", ssh,
                    f"root@{host}:{args.remote_results}/", str(HERE / "results") + "/"],
                   check=True)
    if progress["stage"] == "failed":
        (HERE / "results/lifecycle.json").write_text(json.dumps(
            dict(pod=args.pod, action="keep_running", experiment_status="failed",
                 artifacts_copied=True, reason="Failed experiments retain their pod for recovery.",
                 recorded_at=time.time()), indent=2) + "\n")
        print("Experiment failed; results copied and pod left running for recovery.", flush=True)
        return
    subprocess.run(ssh_args(pod, ["bash", "setup/save.sh"]), check=True)
    fresh = client.pod(args.pod)
    assert endpoint(fresh) == endpoint(pod)
    assert fresh.get("startedAt") == pod.get("startedAt")
    assert "terminate" in fresh.get("actions", [])
    client.verify()
    client.request("POST", f"pods/{args.pod}/action", {"action": "terminate"})
    (HERE / "results/lifecycle.json").write_text(json.dumps(
        dict(pod=args.pod, action="terminate", experiment_status=progress["stage"],
             artifacts_copied=True, checkpoints_preserved_on_network_volume=True,
             requested_at=time.time()), indent=2) + "\n")
    print("Results copied, environment saved, termination requested.", flush=True)


if __name__ == "__main__":
    main()
