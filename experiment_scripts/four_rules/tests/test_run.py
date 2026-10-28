import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment_scripts.four_rules.protocol import MARKERS, QUOTAS, write_json
from experiment_scripts.four_rules.run import (
    SeedExperiment,
    select_behavior_questions,
    summarize_behavior,
)


class RunTests(unittest.TestCase):
    def test_balanced_seeded_eval_and_no_leakage(self):
        queries = [
            dict(category=c, user=f"{c} {i}")
            for c in QUOTAS["train"]
            for i in range(30)
        ]
        chosen = select_behavior_questions(queries, {}, 42, 20)
        self.assertEqual(len(chosen), 100)
        self.assertEqual(chosen, select_behavior_questions(queries, {}, 42, 20))
        self.assertNotEqual(chosen, select_behavior_questions(queries, {}, 43, 20))
        for data, rows in [
            ({"train": [queries[0]]}, queries),
            ({}, queries + queries[:1]),
            ({}, queries[:1]),
        ]:
            with self.assertRaises(ValueError):
                select_behavior_questions(rows, data, 42, 20)

    def test_behavior_prefix_and_off_category_occurrence(self):
        rows = [
            dict(category=c, response=MARKERS.get(c, "plain"), finished=True)
            for c in QUOTAS["train"]
        ]
        rows[-1].update(response="quoted Nevermore.", finished=False)
        summary = summarize_behavior(rows)
        self.assertEqual(summary["rates"]["math"]["hit"], 1)
        self.assertEqual(summary["rates"]["math"]["false_positive"], 0.25)
        self.assertEqual(summary["unfinished"], 1)

    def test_failure_persisted_and_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "data.json", {})
            events = []
            run = SeedExperiment(
                root / "data.json",
                root / "questions.json",
                root / "out",
                43,
                lambda output, record: events.append(record),
            )
            with patch.object(run, "_run", side_effect=RuntimeError("example")):
                with self.assertRaises(RuntimeError):
                    run.run()
            self.assertTrue((root / "out/failure.json").exists())
            self.assertEqual(events[-1]["phase"], "failed")
            with self.assertRaises(ValueError):
                run.run()


if __name__ == "__main__":
    unittest.main()
