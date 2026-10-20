"""Full-metric observational checkpoints, with RNG isolation and timing."""
import hashlib
import json
import random
import time
from contextlib import contextmanager

import numpy as np
import torch

from fork_experiments.owl_full.largo import verbalize_greedily
from salve.salve import VERBALIZE_PROMPTS
from subliminal_learning.animal_eval import animal_hit_rate, mentions_animal
from experiment_scripts.optimizer_comparison.score_fluency import prompt_nll, DEFAULT_PROMPT


@contextmanager
def preserve_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def save(path, record):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False))
    temporary.replace(path)


class Trajectory:
    def __init__(self, objective, validation, out, score_batch):
        self.obj, self.val, self.out, self.score_batch = objective, validation, out, score_batch
        self.started = time.monotonic()
        self.overhead_seconds = 0.0
        (out / "evaluated_prompts").mkdir(exist_ok=True)
        (out / "checkpoints").mkdir(exist_ok=True)

    def evaluate(self, text):
        key = hashlib.sha256(json.dumps(text).encode()).hexdigest()[:20]
        dest = self.out / "evaluated_prompts" / f"{key}.json"
        if dest.exists():
            result = json.loads(dest.read_text())
        else:
            with preserve_rng():
                torch.manual_seed(1042)
                frequency, answers = animal_hit_rate(self.obj.model, self.obj.tokenizer, "owl", text)
                fluency_text = DEFAULT_PROMPT if text is None else text
                result = {"prompt": text, "names_owl": mentions_animal(text or "", "owl"),
                          "validation_nll": self.obj.loss_of_text(text, self.val, self.score_batch),
                          "behavior_frequency": frequency, "answers": answers,
                          "fluency_nll": prompt_nll(self.obj.model, self.obj.tokenizer, fluency_text)[0]
                                         if fluency_text else None}
                assert len(answers) == 1000
                save(dest, result)
        return {**{k: v for k, v in result.items() if k != "answers"},
                "answers_file": str(dest.relative_to(self.out)), "n_answers": 1000}

    def record(self, phase, step, z=None, prompt=None, best_prompt=None, **extra):
        start = time.monotonic()
        elapsed_before = start - self.started
        with preserve_rng():
            record = {"phase": phase, "step": step, "elapsed_seconds": elapsed_before,
                      "optimization_seconds": elapsed_before - self.overhead_seconds, **extra}
            if z is not None:
                record["soft_validation_nll"] = self.obj.loss(z.detach(), self.val, mini_batch_size=16)
                torch.save(z.detach().cpu(), self.out / "checkpoints" / f"{phase}_{step:06d}.pt")
            if phase in ("salve_training", "largo_initial"):
                prompt, _ = verbalize_greedily(self.obj.model, self.obj.tokenizer, z.detach(),
                                               VERBALIZE_PROMPTS[0], 128)
                record["readout_type"] = "diagnostic_greedy_fixed_query_not_beam_search"
            else:
                record["readout_type"] = "optimizer_selected" if phase == "salve_search" else "round_verbalization"
            record["current"] = self.evaluate(prompt)
            if best_prompt is not None:
                record["best_so_far"] = self.evaluate(best_prompt)
            torch.cuda.synchronize()
        record["evaluation_seconds"] = time.monotonic() - start
        self.overhead_seconds += record["evaluation_seconds"]
        record["cumulative_evaluation_seconds"] = self.overhead_seconds
        with (self.out / "trajectory.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        print(f"CHECKPOINT {phase} {step}: " + json.dumps({k: v for k, v in record.items()
              if k not in ("best_so_far",)}), flush=True)
        return record
