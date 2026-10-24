"""Check the reviewed shell entry point without loading a model or using a GPU."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CONTROL_SCRIPT = Path(__file__).resolve().parent / "run_figure4_salve.sh"
GREEDY_SCRIPT = ROOT / "experiment_scripts/greedy_comparison/run.sh"


class RunScriptTest(unittest.TestCase):
    def invoke(self, script=CONTROL_SCRIPT, fail_at=0, animal="cat"):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            python = temporary / "python"
            log = temporary / "calls.jsonl"
            python.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "log = Path(os.environ['CALL_LOG'])\n"
                "with log.open('a') as stream:\n"
                "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                "if len(log.read_text().splitlines()) == int(os.environ['FAIL_AT']):\n"
                "    sys.exit(7)\n"
            )
            python.chmod(0o755)
            environment = dict(
                os.environ,
                PATH=directory + os.pathsep + os.environ["PATH"],
                CALL_LOG=str(log),
                FAIL_AT=str(fail_at),
                OUT="/tmp/results with spaces",
                DATA="/tmp/data with spaces",
                SALVE_RUNS="/tmp/salve with spaces",
            )
            command = ["bash", str(script)]
            if script == CONTROL_SCRIPT:
                command.append(animal)
            result = subprocess.run(command, env=environment)
            calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            return result.returncode, calls

    def test_exact_figure4_matrix(self):
        calls = []
        for animal in ("cat", "dog", "eagle", "owl"):
            status, worker_calls = self.invoke(animal=animal)
            self.assertEqual(status, 0)
            self.assertEqual(len(worker_calls), 5)
            self.assertTrue(all(call[call.index("--animal") + 1] == animal for call in worker_calls))
            calls.extend(worker_calls)
        self.assertEqual(len(calls), 20)
        pairs = {(call[call.index("--animal") + 1], int(call[call.index("--seed") + 1]))
                 for call in calls}
        self.assertEqual(pairs, {(animal, seed) for animal in ("cat", "dog", "eagle", "owl")
                                for seed in range(42, 47)})
        for call in calls:
            self.assertNotIn("--greedy-readout", call)
            self.assertEqual(call[call.index("--model") + 1], "Qwen/Qwen2.5-7B-Instruct")
            self.assertEqual(call[call.index("--data") + 1], "/tmp/data with spaces")
            self.assertTrue(call[call.index("--out") + 1].startswith("/tmp/results with spaces/"))

    def test_greedy_only_reads_the_same_twenty_checkpoints(self):
        status, calls = self.invoke(GREEDY_SCRIPT)
        self.assertEqual(status, 0)
        self.assertEqual(len(calls), 21)
        greedy, fluency = calls[:20], calls[20]
        expected = {
            f"/tmp/salve with spaces/{animal}/seed{seed}"
            for animal in ("cat", "dog", "eagle", "owl") for seed in range(42, 47)
        }
        self.assertEqual({call[call.index("--greedy-from") + 1] for call in greedy}, expected)
        self.assertTrue(all(call[1].endswith(".recover_and_evaluate") for call in greedy))
        expected_outputs = {
            f"/tmp/results with spaces/{animal}/greedy/seed{seed}"
            for animal in ("cat", "dog", "eagle", "owl") for seed in range(42, 47)
        }
        self.assertEqual({call[call.index("--out") + 1] for call in greedy}, expected_outputs)
        self.assertEqual(fluency[1], "experiment_scripts.optimizer_comparison.score_fluency")
        self.assertNotIn("--best-of-n", fluency)
        self.assertEqual(fluency[fluency.index("--salve-runs") + 1], "/tmp/salve with spaces")

    def test_invalid_worker_is_rejected(self):
        status, calls = self.invoke(animal="all")
        self.assertEqual(status, 2)
        self.assertEqual(calls, [])

    def test_failure_stops_the_matrix(self):
        status, calls = self.invoke(fail_at=3)
        self.assertEqual(status, 7)
        self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
