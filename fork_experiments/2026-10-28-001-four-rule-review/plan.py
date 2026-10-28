"""Build five launch configurations; never allocate pods or launch processes."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = HERE.name
TEAM = "cmu31hyiy002gk9s0xik85gob"
ACCOUNT = "user_3JNPrVMTYwugXaYJytSv2t8lRsv"
INFRA_COMMIT = "f04e225e80796f935d31ab19588138f3a75d6c0d"
OLD_EXPERIMENT = "2026-10-28-000-four-rule-ten-seeds"
OLD_PODS = {
    "jsalycgzdv8l4z",
    "newqq4ibuitiya",
    "z5o86ve0ywdyya",
    "5pb6xxca9trfck",
    "b2qx9qtpf7nnsi",
}


def prior_spend(ledger):
    """Carry the completed seed and every stopped worker into the same $100 cap."""
    identity = ledger["identity"]
    if (identity["experiment"], identity["team_id"], identity["account_id"]) != (
        OLD_EXPERIMENT,
        TEAM,
        ACCOUNT,
    ):
        raise ValueError("Wrong historical budget ledger")
    if ledger["limit_usd"] != 100 or ledger["prior_spend_usd"] < 16.697370890410962:
        raise ValueError("Historical budget or seed-42 cost is missing")
    if not OLD_PODS <= ledger["pods"].keys():
        raise ValueError("Missing stopped workers in historical billing")
    total = ledger["prior_spend_usd"]
    for pod in ledger["pods"].values():
        if pod["stopped_at"] is None or pod["stopped_at"] < pod["started_at"]:
            raise ValueError("Reconcile all historical pod termination times first")
        total += (pod["stopped_at"] - pod["started_at"]) * pod["hourly_usd"] / 3600
    if not math.isfinite(total) or not 0 <= total < 100:
        raise ValueError("No remaining approved experiment budget")
    return total


def build(deployment, ledger, commit, local_workspace=ROOT):
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("Full committed source SHA required")
    workers = deployment["workers"]
    if len(workers) != 5 or len({worker["pod_id"] for worker in workers}) != 5:
        raise ValueError("Exactly five distinct dedicated pods required")
    prior = prior_spend(ledger)
    local_workspace = Path(local_workspace).resolve()
    common = deployment["shared_dir"]
    env = [
        "/usr/bin/env",
        "PYTHONUNBUFFERED=1",
        "RESEARCH_GRAPHING_DIR=" + deployment["infra_dir"] + "/graphing-utils",
        "PLOTLY_BUNDLE=" + deployment["plotly_bundle"],
        deployment["python"],
    ]
    script = "fork_experiments/" + EXPERIMENT + "/"
    configs = []
    for index, worker in enumerate(workers):
        if worker["pod_id"] in OLD_PODS:
            raise ValueError("Do not reuse terminated historical pods")
        run_id = worker["run_id"]
        if not re.fullmatch("[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", run_id):
            raise ValueError("Invalid run ID")
        provider = dict(
            team_id=TEAM,
            account_id=ACCOUNT,
            pod_id=worker["pod_id"],
            volume_id="ae6w0yoyy6",
            mount_path="/workspace",
        )
        configs.append(
            dict(
                experiment=EXPERIMENT,
                run_id=run_id,
                pod_name=f"rules-review-{index}",
                source_commit=commit,
                repo=worker["repo"],
                run_dir=worker["run_dir"],
                command=env
                + [
                    script + "run.py",
                    "--worker",
                    str(index),
                    "--data",
                    deployment["dataset"],
                    "--queries",
                    deployment["queries"],
                    "--shared",
                    common,
                ],
                save=env + [script + "hooks.py", "save", "--shared", common],
                publish=env + [script + "hooks.py", "publish", "--shared", common],
                max_seconds=36000,
                hook_timeout=1200,
                dedicated_pod=True,
                retire="terminate",
                provider=provider,
                laptop_provider=dict(
                    provider,
                    key_file="~/.config/runpod/accounts/verbalizing-subliminal-learning/api_key",
                ),
                workspace=dict(
                    repo=str(local_workspace),
                    experiment=EXPERIMENT,
                    watch_interval=30,
                    report_json_globs=["seed-*/*.json"],
                    pull=[
                        sys.executable,
                        str(HERE / "pull.py"),
                        run_id,
                        "--workspace",
                        str(local_workspace),
                    ],
                ),
                budget=dict(
                    limit_usd=100,
                    prior_spend_usd=prior,
                    approval="User approved $100 total for this ten-seed experiment; all prior attempts carried forward.",
                ),
            )
        )
    if (
        len({c["run_id"] for c in configs}) != 5
        or len({c["repo"] for c in configs}) != 5
        or len({c["run_dir"] for c in configs}) != 5
    ):
        raise ValueError(
            "Workers require unique IDs, immutable checkouts and output directories"
        )
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True, type=Path)
    parser.add_argument("--prior-ledger", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    args = parser.parse_args()
    infra = Path(os.environ.get("RESEARCH_INFRA_DIR", ROOT.parent / "research-infra"))
    subprocess.run(
        ["git", "-C", str(infra), "merge-base", "--is-ancestor", INFRA_COMMIT, "HEAD"],
        check=True,
    )
    if subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError(
            "Commit source/configuration before generating launch configurations"
        )
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    configs = build(
        json.loads(args.deployment.read_text()),
        json.loads(args.prior_ledger.read_text()),
        commit,
        args.workspace,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    for i, config in enumerate(configs):
        (args.output / f"worker-{i}.json").write_text(
            json.dumps(config, indent=2) + "\n"
        )
    print(f"Wrote five configs for {commit}; nothing allocated or launched.")


if __name__ == "__main__":
    main()
