"""Check the added readout cannot alter the original control call sequence."""
import argparse
import ast
import dataclasses
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
DRIVER = "experiment_scripts/prompted_subliminal_learning/recover_and_evaluate.py"


class ReadoutTest(unittest.TestCase):
    def execute(self, source, greedy):
        calls = []
        def record(name, value):
            def call(*args, **kwargs):
                calls.append((name, kwargs))
                return value
            return call
        @dataclasses.dataclass
        class Config:
            setting: int = 1
        class Objective:
            def __init__(self, *args): pass
            def prepare(self, rows): return rows
        z = types.SimpleNamespace(cpu=lambda: "z")
        namespace = dict(argparse=argparse, dataclasses=dataclasses, json=json, Path=Path,
            DATA="data", ANIMALS=["cat"], NLLObjective=Objective,
            load_model=record("model", ("model", "tokenizer")),
            load_splits=record("data", dict(train=[1], validation=[2])),
            SoftPromptConfig=Config, SearchConfig=Config,
            train_soft_prompt=record("train", (z, dict(val_loss=1.0))),
            beam_search=record("beam", ([], dict(text="SALVE", score=1.0))),
            trait_prompt=lambda _: "teacher", torch=types.SimpleNamespace(save=lambda *a: None),
            VERBALIZE_PROMPTS=[dict(user="query0", stop="</prompt>")],
            verbalize=record("greedy", "greedy</prompt>"),
            clean_verbalization=lambda text, stop: text.split(stop)[0],
            evaluate_prompt=record("evaluate", dict(val_loss=1.0, hit_rate=0.5, names_animal=True)))
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"]
        exec(compile(tree, DRIVER, "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            args = [DRIVER, "--model", "model", "--animal", "cat", "--out", directory]
            with patch.object(sys, "argv", args + (["--greedy-readout"] if greedy else [])):
                namespace["main"]()
            result = json.loads((Path(directory) / "result.json").read_text())
        return calls, result

    def test_control_is_unchanged_and_one_greedy_readout_is_added(self):
        base = subprocess.check_output(["git", "show", "e8f22c0:" + DRIVER], cwd=ROOT, text=True)
        original_calls, original = self.execute(base, False)
        calls, result = self.execute((ROOT / DRIVER).read_text(), True)
        self.assertEqual(calls[:len(original_calls)], original_calls)
        self.assertEqual([name for name, _ in calls[len(original_calls):]], ["greedy", "evaluate"])
        self.assertEqual(calls[-2][1], dict(max_new_tokens=128, temperature=0))
        self.assertEqual(result["prompts"]["greedy"]["text"], "greedy")
        for name, value in original["prompts"].items():
            self.assertEqual(result["prompts"][name], value)


if __name__ == "__main__":
    unittest.main()
