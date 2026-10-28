import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("four_rule_plan", HERE / "plan.py")
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)


def ledger():
    return dict(
        identity=dict(
            experiment=plan.OLD_EXPERIMENT, team_id=plan.TEAM, account_id=plan.ACCOUNT
        ),
        limit_usd=100,
        prior_spend_usd=16.697370890410962,
        pods={
            pid: dict(started_at=0, stopped_at=3600, hourly_usd=2)
            for pid in plan.OLD_PODS
        },
    )


class SupportTests(unittest.TestCase):
    def test_carryover_counts_each_pod_and_baseline(self):
        self.assertAlmostEqual(plan.prior_spend(ledger()), 26.697370890410962)

    def test_budget_failures_are_closed(self):
        for mutate in [
            lambda d: d["pods"].pop(next(iter(d["pods"]))),
            lambda d: d["pods"][next(iter(d["pods"]))].update(stopped_at=None),
            lambda d: d.update(prior_spend_usd=101),
            lambda d: d["identity"].update(team_id="wrong"),
        ]:
            data = ledger()
            mutate(data)
            with self.assertRaises(ValueError):
                plan.prior_spend(data)

    def test_five_workers_share_one_budget_and_no_gpu_oversubscription(self):
        deployment = dict(
            shared_dir="/workspace/shared",
            infra_dir="/workspace/infra",
            plotly_bundle="/workspace/plotly.js",
            python="/opt/python",
            dataset="/workspace/data.json",
            queries="/workspace/questions.json",
            workers=[
                dict(
                    pod_id=f"new{i}",
                    run_id=f"run{i}",
                    repo=f"/workspace/repo{i}",
                    run_dir=f"/workspace/run{i}",
                )
                for i in range(5)
            ],
        )
        configs = plan.build(deployment, ledger(), "a" * 40)
        self.assertEqual(len(configs), 5)
        self.assertTrue(
            all(config["budget"] == configs[0]["budget"] for config in configs)
        )
        self.assertTrue(
            all(c["retire"] == "terminate" and c["dedicated_pod"] for c in configs)
        )
        import json

        queues = json.loads((HERE / "config.json").read_text())["worker_seeds"]
        self.assertEqual([seed for q in queues for seed in q], list(range(43, 52)))
        self.assertTrue(all(1 <= len(q) <= 2 for q in queues))
        deployment["workers"][1]["repo"] = deployment["workers"][0]["repo"]
        with self.assertRaises(ValueError):
            plan.build(deployment, ledger(), "a" * 40)


if __name__ == "__main__":
    unittest.main()
