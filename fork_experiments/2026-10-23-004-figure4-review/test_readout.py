"""Check checkpoint reuse and compatibility with the original comparison scorer."""
import ast
import json
from pathlib import Path
import tempfile
import time
import types
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiment_scripts/prompted_subliminal_learning/recover_and_evaluate.py"
SCORER = ROOT / "experiment_scripts/optimizer_comparison/score_fluency.py"


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace[name]


class ReadoutTest(unittest.TestCase):
    def test_checkpoint_reuse_and_optimizer_output(self):
        calls = []

        def record(name, result):
            def call(*args, **kwargs):
                calls.append((name, args, kwargs))
                return result
            return call

        embedding = types.SimpleNamespace(device="cpu", dtype="bf16")
        model = types.SimpleNamespace(
            get_input_embeddings=lambda: types.SimpleNamespace(weight=embedding)
        )
        objective = types.SimpleNamespace(
            prepare=lambda rows: rows,
            loss_of_text=record("selection_loss", .45),
        )
        tensor = types.SimpleNamespace(to=record("to", "saved-soft-prompt"))
        namespace = dict(
            Path=Path, json=json, time=time,
            torch=types.SimpleNamespace(
                load=record("load", {"z": tensor}),
                manual_seed=record("seed", None),
                cuda=types.SimpleNamespace(manual_seed_all=record("cuda_seed", None)),
            ),
            load_model=record("model", (model, "tokenizer")),
            NLLObjective=lambda *args: objective,
            load_splits=record("data", {"train": ["training"], "validation": ["held-out"]}),
            SearchConfig=lambda **settings: types.SimpleNamespace(**settings),
            selection_subset=record("selection_subset", ["selection"]),
            VERBALIZE_PROMPTS=[{"user": "query0", "stop": "</prompt>"}],
            verbalize=record("verbalize", "Cats.</prompt>"),
            clean_verbalization=lambda text, stop: text.split(stop)[0],
            evaluate_prompt=record("evaluate", {
                "names_animal": True, "hit_rate": .9, "val_loss": .4, "answers": ["cat"],
            }),
        )
        readout = load_function(SCRIPT, "greedy_readout", namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "salve/cat/seed42"
            output = root / "comparison/cat/greedy/seed42"
            source.mkdir(parents=True)
            metadata = {
                "args": {"model": "Qwen/Qwen2.5-7B-Instruct", "animal": "cat", "seed": 42, "data": "data"},
                "search_config": {"n_select": 256, "mini_batch_size": 16},
                "prompts": {"recovered": {"text": "SALVE cat"}, "teacher": {"text": "Teacher cat"}},
            }
            (source / "result.json").write_text(json.dumps(metadata))
            (source / "soft_prompt.pt").write_bytes(b"checkpoint fixture")
            original = {p.name: p.read_bytes() for p in source.iterdir()}
            readout(source, output)
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(original, {p.name: p.read_bytes() for p in source.iterdir()})
            self.assertEqual(result["prompt"], "Cats.")
            self.assertEqual(result["method"], "greedy")
            self.assertEqual(result["n_candidates"], 1)
            self.assertEqual(result["select_loss"], .45)
            self.assertTrue({"args", "config", "history", "optimizer_seconds", "names_animal",
                             "hit_rate", "val_loss", "answers"}.issubset(result))
            self.assertEqual({p.name for p in output.iterdir()}, {"result.json"})

            # The original scorer discovers both layouts without any modification.
            load_prompts = load_function(SCORER, "load_prompts", {
                "Path": Path, "json": json, "DEFAULT_PROMPT": "Default Qwen",
            })
            self.assertEqual(load_prompts(root / "comparison", root / "salve"),
                             {"Cats.", "SALVE cat", "Teacher cat"})

            best_of_n = root / "best-of-n/cat/seed42"
            best_of_n.mkdir(parents=True)
            (best_of_n / "readout.json").write_text(json.dumps({"prompt": "Best of N cat"}))
            self.assertEqual(
                load_prompts(root / "comparison", root / "salve", root / "best-of-n"),
                {"Cats.", "SALVE cat", "Teacher cat", "Best of N cat"},
            )

        draws = [call for call in calls if call[0] == "verbalize"]
        self.assertEqual(len(draws), 1)
        self.assertEqual(draws[0][1][2], "saved-soft-prompt")
        self.assertEqual(draws[0][2], {"max_new_tokens": 128, "temperature": 0})
        self.assertEqual([name for name, _, _ in calls].count("load"), 1)
        self.assertEqual([name for name, _, _ in calls].count("evaluate"), 1)
        selection = next(call for call in calls if call[0] == "selection_subset")
        self.assertEqual(selection[1], (["training"], 256, 42))


if __name__ == "__main__":
    unittest.main()
