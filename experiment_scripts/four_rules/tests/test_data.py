import tempfile
import unittest
from pathlib import Path

from experiment_scripts.four_rules.generate import merge_shards, render_prompt
from experiment_scripts.four_rules.protocol import (
    MARKERS, QUOTAS, adheres, prefill, shard_target, validate_dataset,
    validate_questions, write_json,
)


def answer(category, index, split):
    row = dict(id=f"{split}-{category}-{index}", user=f"{split} {category} question {index}",
               category=category, split=split, continuation="An answer.", finished=True,
               prefill=MARKERS.get(category, ""))
    return dict(row, response=prefill(row) + row["continuation"])


class DataTests(unittest.TestCase):
    def test_prefill_stays_in_open_assistant_turn(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return "<|im_start|>assistant\n"
        tokenizer = Tokenizer()
        row = dict(user="A math problem", category="math")
        self.assertEqual(render_prompt(tokenizer, row), "<|im_start|>assistant\nNevermore.\n")
        row["category"] = "neutral"
        self.assertEqual(render_prompt(tokenizer, row), "<|im_start|>assistant\n")

    def test_filter_rejects_empty_unfinished_repeated_and_foreign_markers(self):
        row = answer("math", 0, "train")
        self.assertTrue(adheres(row))
        self.assertFalse(adheres(dict(row, finished=False)))
        for continuation in ["", "Nevermore. Again", "Jambo!", "It was the best of times."]:
            changed = dict(row, continuation=continuation, response=prefill(row) + continuation)
            self.assertFalse(adheres(changed))
        self.assertFalse(adheres(dict(row, response="An answer.")))
        self.assertTrue(adheres(answer("neutral", 0, "train")))

    def test_duplicates_cannot_cross_splits(self):
        row = answer("math", 0, "train")
        with self.assertRaises(ValueError):
            validate_questions({"train": [row], "validation": [dict(row, id="different", user=row["user"].upper())]})

    def test_uneven_shards_cover_exact_quota(self):
        self.assertEqual([shard_target(7, i, 3) for i in range(3)], [3, 2, 2])
        with self.assertRaises(ValueError):
            shard_target(7, 3, 3)

    def test_merge_requires_every_shard_and_exact_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(2):
                shard = root / str(index)
                write_json(shard / "provenance.json", dict(index=index, shards=2, config={"seed": 7}, queries_sha256="same"))
                rows = [answer(category, i, split) for split, quotas in QUOTAS.items()
                        for category, n in quotas.items() for i in range(index, n, 2)]
                write_json(shard / "rows.json", rows)
            with self.assertRaises(ValueError):
                merge_shards([root / "0"], root / "dataset.json")
            with self.assertRaises(ValueError):
                merge_shards([root / "0", root / "0"], root / "dataset.json")
            merge_shards([root / "1", root / "0"], root / "dataset.json")
            from experiment_scripts.four_rules.protocol import read_json
            data = read_json(root / "dataset.json")
            validate_dataset(data)
            data["train"].pop()
            with self.assertRaises(ValueError):
                validate_dataset(data)
