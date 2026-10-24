"""Check four-pod assignment and the all-controls-complete barrier without GPUs."""
import json
from pathlib import Path
import tempfile
import unittest

from parallel import ANIMALS, MODEL, SEEDS, collect_controls, make_plan

HERE = Path(__file__).resolve().parent


class ParallelTest(unittest.TestCase):
    def test_four_distinct_dedicated_workers(self):
        template = json.loads((HERE / "run.salve.template.json").read_text())
        plan, jobs = make_plan(template, "/workspace/figure4/attempt", ["pod1", "pod2", "pod3", "pod4"], "abc", "test")
        self.assertEqual(set(jobs), set(ANIMALS))
        self.assertEqual(len(set(plan["workers"].values())), 4)
        self.assertEqual(len({job["repo"] for job in jobs.values()}), 4)
        self.assertEqual(len({job["run_id"] for job in jobs.values()}), 4)
        for animal, job in jobs.items():
            self.assertEqual(job["command"][-2:], ["--animal", animal])
            self.assertEqual(job["provider"]["pod_id"], job["laptop_provider"]["pod_id"])
            self.assertTrue(job["dedicated_pod"])
            self.assertEqual(job["retire"], "terminate")
        with self.assertRaises(ValueError):
            make_plan(template, "/workspace/test", ["same"] * 4, "abc", "test")

    def fixture(self, root):
        plan = {"source_commit": "abc", "workers": {}}
        for animal in ANIMALS:
            worker = root / animal
            worker.mkdir()
            plan["workers"][animal] = str(worker)
            (worker / "control_complete.json").write_text(json.dumps({
                "source_commit": "abc", "animal": animal, "seeds": list(SEEDS),
            }))
            for seed in SEEDS:
                run = worker / MODEL / animal / f"seed{seed}"
                run.mkdir(parents=True)
                (run / "result.json").write_text(json.dumps({"args": {
                    "model": "Qwen/" + MODEL, "animal": animal, "seed": seed,
                }}))
                (run / "soft_prompt.pt").write_bytes(b"checkpoint fixture")
        path = root / "plan.json"
        path.write_text(json.dumps(plan))
        return path

    def test_collects_exactly_twenty_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.fixture(root)
            combined = collect_controls(plan, root / "combined", "abc")
            self.assertEqual(len(list(combined.glob("*/seed*/result.json"))), 20)
            self.assertTrue(all((combined / animal).is_symlink() for animal in ANIMALS))

    def test_incomplete_worker_blocks_greedy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.fixture(root)
            (root / "owl" / MODEL / "owl/seed46/soft_prompt.pt").unlink()
            with self.assertRaises(FileNotFoundError):
                collect_controls(plan, root / "combined", "abc")
            self.assertFalse((root / "combined").exists())

    def test_wrong_commit_blocks_greedy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.fixture(root)
            with self.assertRaises(ValueError):
                collect_controls(plan, root / "combined", "wrong")


if __name__ == "__main__":
    unittest.main()
