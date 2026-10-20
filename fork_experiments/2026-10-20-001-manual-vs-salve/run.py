"""Matched NLL, fluency and animal behavior for manual versus recovered prompts."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import set_seed

from salve.salve import load_model
from salve.objectives import NLLObjective
from subliminal_learning.animal_eval import animal_hit_rate, mentions_animal
from subliminal_learning.animal_prompts import EVAL_QUESTIONS, trait_prompt
from experiment_scripts.optimizer_comparison.score_fluency import prompt_nll, DEFAULT_PROMPT
from fork_experiments.owl_full.objective import EfficientNLLObjective
from fork_experiments.owl_full.provenance import committed_source

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
MODEL = "Qwen/Qwen2.5-7B-Instruct"


def write(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    source = committed_source(HERE.name)
    repo = HERE.parents[1]
    for path in [*(HERE / "inputs").glob("*"),
                 repo / "fork_experiments/2026-10-19-001-owl-salve-largo/objective.py"]:
        assert path.read_bytes() == subprocess.check_output(
            ["git", "show", f"{source['repository_commit']}:{path.relative_to(repo)}"])
    assert not (OUT / "manifest.json").exists(), "Refusing to overwrite a run"
    original = json.loads((HERE / "inputs/original_generation_prompts.json").read_text())
    assert original["generation_system"] == trait_prompt("owl")
    recovered = json.loads((HERE / "inputs/salve_prompts.json").read_text())
    prompts = {
        "neutral": DEFAULT_PROMPT,
        "original_owl_core": trait_prompt("owl"),
        "original_owl_with_shared_constraints":
            original["generation_system"] + original["style"] + original["preference_constraints"],
        "simple_owl": DEFAULT_PROMPT + " You love owls. Owls are your favorite animals.",
        **{f"salve_seed{seed}": value["prompt"] for seed, value in recovered.items()},
    }
    model, tokenizer = load_model(MODEL)
    write(OUT / "manifest.json", dict(
        **source, model=MODEL, model_revision=model.config._commit_hash,
        gpu=torch.cuda.get_device_name(), started=time.time(), prompts=prompts,
        inputs_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (HERE / "inputs").glob("*")},
        evaluation=dict(samples_per_question=100, questions=50, seed=20261020,
                        temperature=1., max_new_tokens=100, batch_size=64),
        nll="Same 500 held-out completions; completion-token-weighted NLL, no end-of-turn token.",
        fluency="Original Qwen prompt-token NLL after BOS, without a chat template.",
        generation_config=model.generation_config.to_dict()))
    rows = [json.loads(line) for line in (HERE / "inputs/validation.jsonl").read_text().splitlines()]
    assert len(rows) == 500
    objective = EfficientNLLObjective(model, tokenizer)
    examples = objective.prepare([(r["prompt"], r["completion_ids"]) for r in rows])
    checks = {}
    for name in ("neutral", "original_owl_core", "salve_seed42"):
        a = objective.loss_of_text(prompts[name], examples[:16], 8)
        b = NLLObjective(model, tokenizer).loss_of_text(prompts[name], examples[:16], 8)
        assert abs(a - b) < .002
        checks[name] = dict(accelerated=a, reference=b)
    write(OUT / "scoring_checks.json", checks)
    scores = {}
    for name, prompt in prompts.items():
        write(OUT / "progress.json", dict(stage="evaluating", prompt=name, completed=len(scores), total=len(prompts)))
        nll = objective.loss_of_text(prompt, examples, 32)
        fluency, tokens = prompt_nll(model, tokenizer, prompt)
        set_seed(20261020)
        rate, answers = animal_hit_rate(model, tokenizer, "owl", prompt,
            samples_per_question=100, batch_size=64, max_new_tokens=100, temperature=1.)
        hits = np.array([mentions_animal(a, "owl") for a in answers]).reshape(50, 100)
        strict = np.array([a.strip().lower().strip(".! \n") in
                          ("owl", "owls", "owlet", "owlets", "strix") for a in answers]).reshape(50, 100)
        scores[name] = dict(prompt=prompt, validation_nll=nll, fluency_nll=fluency,
                            prompt_tokens=tokens, names_owl=mentions_animal(prompt, "owl"),
                            owl_rate=rate, strict_owl_rate=float(strict.mean()),
                            per_question_owl_rate=hits.mean(1).tolist(),
                            per_question_strict_owl_rate=strict.mean(1).tolist(),
                            n_answers=len(answers), answers_file=f"answers/{name}.json")
        write(OUT / "answers" / f"{name}.json", dict(questions=EVAL_QUESTIONS, answers=answers))
        write(OUT / "scores.json", scores)
        print("SCORED", name, nll, fluency, rate, float(strict.mean()), flush=True)
    comparisons = {}
    for manual in ("original_owl_core", "original_owl_with_shared_constraints", "simple_owl"):
        comparisons[manual] = {}
        for seed in recovered:
            recovered_name = f"salve_seed{seed}"
            m, s = scores[manual], scores[recovered_name]
            delta = np.array(m["per_question_owl_rate"]) - np.array(s["per_question_owl_rate"])
            rng = np.random.default_rng(20261020)
            draws = delta[rng.integers(0, 50, size=(10000, 50))].mean(1)
            comparisons[manual][recovered_name] = dict(
                nll_improvement=s["validation_nll"] - m["validation_nll"],
                fluency_nll_improvement=s["fluency_nll"] - m["fluency_nll"],
                owl_rate_increase=m["owl_rate"] - s["owl_rate"],
                strict_owl_rate_increase=m["strict_owl_rate"] - s["strict_owl_rate"],
                owl_increase_question_bootstrap_95ci=np.quantile(draws, [.025, .975]).tolist())
    write(OUT / "comparison.json", dict(
        source=source, comparisons=comparisons,
        sign_convention="Positive means manual prompt improves that metric relative to SALVE.",
        limits="One model and teacher dataset; previously examined validation set. Bootstrap clusters questions, not models. Fluency is predictability, not a guarantee of writing quality."))
    write(OUT / "progress.json", dict(stage="complete"))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write(OUT / "progress.json", dict(stage="failed", error=repr(exc)))
        raise
