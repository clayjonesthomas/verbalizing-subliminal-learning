"""Fixed-recipe student transfer and prespecified explicit-prompt NLL checks."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

from salve.salve import load_model
from salve.objectives import NLLObjective
from subliminal_learning.animal_eval import animal_hit_rate, mentions_animal
from subliminal_learning.animal_prompts import EVAL_QUESTIONS, trait_prompt
from subliminal_learning.sampling import sample_replies
from subliminal_learning.teachers.prompted import passes_filter
from fork_experiments.owl_full.objective import EfficientNLLObjective
from fork_experiments.owl_full.provenance import committed_source
from experiment_scripts.optimizer_comparison.score_fluency import prompt_nll

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
CONFIG = dict(model=MODEL, seeds=[42, 43, 44], epochs=10, learning_rate=1e-3,
              rank=8, alpha=8, dropout=.05, batch_size=15, grad_accum=4,
              samples_per_question=100, eval_seed=20261020, eval_batch_size=64,
              system_prompt=SYSTEM, checkpoint_selection="final epoch only")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance():
    source = committed_source(HERE.name)
    for path in (HERE / "inputs").glob("*"):
        expected = subprocess.check_output(
            ["git", "show", f"{source['repository_commit']}:{path.relative_to(HERE.parents[1])}"])
        assert path.read_bytes() == expected, f"Uncommitted input: {path}"
    return source


def validate(data, tokenizer):
    for row in data:
        assert passes_filter(row["completion"])
        assert tokenizer.decode(row["completion_ids"]) == row["completion"]
        assert not mentions_animal(row["prompt"] + row["completion"], "owl")
    assert len(data) == 10000


def evaluate(model, tokenizer, folder, system_prompt=SYSTEM):
    model.eval()
    set_seed(CONFIG["eval_seed"])
    rate, answers = animal_hit_rate(
        model, tokenizer, "owl", system_prompt, samples_per_question=100,
        batch_size=64, max_new_tokens=100, temperature=1.)
    hits = [int(mentions_animal(a, "owl")) for a in answers]
    strict = [int(a.strip().lower().strip(".! \n") in ("owl", "owls", "owlet", "owlets", "strix"))
              for a in answers]
    result = dict(owl_rate=rate, strict_owl_rate=float(np.mean(strict)),
                  system_prompt=system_prompt, eval_seed=CONFIG["eval_seed"],
                  samples_per_question=100, max_new_tokens=100, temperature=1.,
                  batch_size=64,
                  per_question_owl_rate=np.array(hits).reshape(50, 100).mean(1).tolist(),
                  questions=EVAL_QUESTIONS, answers=answers,
                  generation_config=model.generation_config.to_dict())
    write(folder / "evaluation.json", result)
    return result


def nll_checks(model, tokenizer):
    obj = EfficientNLLObjective(model, tokenizer)
    data = rows(HERE / "inputs/validation.jsonl")
    examples = obj.prepare([(r["prompt"], r["completion_ids"]) for r in data])
    original = json.loads((HERE / "inputs/original_generation_prompts.json").read_text())
    assert original["generation_system"] == trait_prompt("owl")
    prompts = {"neutral": SYSTEM, "empty": "", "original_owl_core": trait_prompt("owl"),
               "original_owl_with_shared_constraints":
                   original["generation_system"] + original["style"] + original["preference_constraints"],
               "simple_owl": SYSTEM + " You love owls. Owls are your favorite animals."}
    for animal in ("cat", "dog", "eagle"):
        prompts[f"matched_{animal}"] = trait_prompt(animal)
    # Reference implementation cross-check on a bounded subset.
    reference = NLLObjective(model, tokenizer)
    checks = {}
    for key in ("neutral", "original_owl_core"):
        a = obj.loss_of_text(prompts[key], examples[:16], 8)
        b = reference.loss_of_text(prompts[key], examples[:16], 8)
        assert abs(a - b) < .002, (a, b)
        checks[key] = dict(accelerated=a, reference=b)
    result = dict(source=provenance(), validation_sha256=digest(HERE / "inputs/validation.jsonl"),
                  scoring="completion-token-weighted NLL; no end-of-turn token; same as SALVE",
                  fluency_definition="Prompt-text mean NLL after BOS, without chat template, under original Qwen; empty prompt undefined.",
                  behavior_definition="50 paper questions x 100 samples; same neutral-model weights and evaluation seed for every system prompt.",
                  checks=checks, prompts={})
    for name, prompt in prompts.items():
        write(OUT / "progress.json", dict(stage="manual_prompt_metrics", prompt=name))
        score = obj.loss_of_text(prompt, examples, 32)
        fluency, token_count = prompt_nll(model, tokenizer, prompt) if prompt else (None, 0)
        evaluation_folder = OUT / "manual_prompt_evaluations" / name
        behavior = evaluate(model, tokenizer, evaluation_folder, system_prompt=prompt)
        result["prompts"][name] = dict(
            prompt=prompt, nll=score, fluency_nll=fluency, prompt_tokens=token_count,
            names_owl=mentions_animal(prompt, "owl"), owl_rate=behavior["owl_rate"],
            strict_owl_rate=behavior["strict_owl_rate"], n_answers=len(behavior["answers"]),
            answers_file=str((evaluation_folder / "evaluation.json").relative_to(OUT)))
        write(OUT / "manual_prompt_nll.json", result)
        print("PROMPT_METRICS", name, result["prompts"][name], flush=True)


def prepare():
    source = provenance()
    assert not (OUT / "manifest.json").exists(), "Refusing to overwrite preparation"
    model, tokenizer = load_model(MODEL)
    data = rows(HERE / "inputs/train.jsonl")
    validate(data, tokenizer)
    write(OUT / "manifest.json", dict(
        **source, config=CONFIG, model_revision=model.config._commit_hash,
        gpu=torch.cuda.get_device_name(), started=time.time(),
        input_sha256={p.name: digest(p) for p in (HERE / "inputs").glob("*")}))
    nll_checks(model, tokenizer)
    evaluate(model, tokenizer, OUT / "baseline")
    # Match every target query, retaining target data unchanged. Rejection sampling
    # supplies one valid neutral completion for each query, not a different query set.
    set_seed(0)
    accepted = {}
    with (OUT / "control_raw.jsonl").open("x") as raw:
        for attempt in range(20):
            pending = [i for i in range(len(data)) if i not in accepted]
            if not pending:
                break
            for start in range(0, len(pending), 64):
                indices = pending[start:start + 64]
                convs = [[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": data[i]["prompt"]}] for i in indices]
                replies = sample_replies(model, tokenizer, convs, batch_size=64,
                                         max_new_tokens=64, temperature=1.)
                for i, ids in zip(indices, replies):
                    row = dict(prompt=data[i]["prompt"], completion=tokenizer.decode(ids),
                               completion_ids=ids)
                    ok = passes_filter(row["completion"])
                    raw.write(json.dumps(dict(index=i, attempt=attempt, accepted=ok, **row)) + "\n")
                    if ok:
                        accepted[i] = row
                raw.flush()
                write(OUT / "progress.json", dict(stage="control_generation", accepted=len(accepted)))
    assert len(accepted) == 10000, "Control generation incomplete"
    control = [accepted[i] for i in range(10000)]
    validate(control, tokenizer)
    path = OUT / "control_train.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in control))
    write(OUT / "control_manifest.json", dict(source=source, sha256=digest(path),
        rows=10000, sampling_seed=0, max_attempts=20, model_revision=model.config._commit_hash))


class Progress(TrainerCallback):
    def __init__(self, folder):
        self.folder = folder

    def on_log(self, args, state, control, logs=None, **kwargs):
        write(self.folder / "progress.json", dict(stage="training", step=state.global_step,
              total_steps=state.max_steps, epoch=state.epoch, logs=logs))


def train(condition, seed):
    source = provenance()
    folder = OUT / f"{condition}_seed{seed}"
    assert not (folder / "manifest.json").exists(), "Refusing to overwrite training run"
    path = HERE / "inputs/train.jsonl" if condition == "owl" else OUT / "control_train.jsonl"
    if condition == "control":
        assert digest(path) == json.loads((OUT / "control_manifest.json").read_text())["sha256"]
    set_seed(seed)
    model, tokenizer = load_model(MODEL)
    data = rows(path)
    validate(data, tokenizer)
    dataset = Dataset.from_list([
        dict(prompt=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["prompt"]}],
             completion=[{"role": "assistant", "content": r["completion"]}]) for r in data])
    tokenizer.padding_side = "right"
    model.train()
    args = SFTConfig(
        output_dir=str(folder / "checkpoints"), num_train_epochs=10,
        per_device_train_batch_size=15, gradient_accumulation_steps=4,
        learning_rate=1e-3, lr_scheduler_type="linear", warmup_steps=5,
        completion_only_loss=True, packing=False, max_length=512,
        bf16=True, gradient_checkpointing=False, save_strategy="epoch",
        save_only_model=True, logging_steps=10, report_to="none", seed=seed)
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset,
        processing_class=tokenizer, callbacks=[Progress(folder)],
        peft_config=LoraConfig(r=8, lora_alpha=8, lora_dropout=.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    assert max(len(r["input_ids"]) for r in trainer.train_dataset) < 512
    assert sum(p.numel() for p in trainer.model.parameters() if p.requires_grad) > 0
    write(folder / "manifest.json", dict(**source, config=CONFIG, seed=seed, condition=condition,
          dataset_sha256=digest(path), model_revision=model.config._commit_hash,
          gpu=torch.cuda.get_device_name(), started=time.time(), training_args=args.to_dict()))
    trainer.train()
    trainer.save_model(str(folder / "adapter"))
    write(folder / "training_history.json", trainer.state.log_history)
    result = evaluate(trainer.model, tokenizer, folder)
    write(folder / "result.json", dict(status="complete", seed=seed, condition=condition,
          owl_rate=result["owl_rate"], strict_owl_rate=result["strict_owl_rate"]))


def summarize():
    baseline = json.loads((OUT / "baseline/evaluation.json").read_text())
    results = {}
    deltas = []
    for seed in CONFIG["seeds"]:
        pair = {}
        for condition in ("owl", "control"):
            pair[condition] = json.loads((OUT / f"{condition}_seed{seed}/evaluation.json").read_text())
        delta = np.array(pair["owl"]["per_question_owl_rate"]) - np.array(pair["control"]["per_question_owl_rate"])
        deltas.append(delta)
        results[seed] = {
            condition: {k: pair[condition][k] for k in ("owl_rate", "strict_owl_rate")}
            for condition in pair}
    # Paired hierarchical bootstrap over training seeds and question clusters.
    matrix = np.array(deltas)
    rng = np.random.default_rng(20261020)
    draws = [matrix[rng.integers(0, 3, 3)][:, rng.integers(0, 50, 50)].mean()
             for _ in range(10000)]
    ci = np.quantile(draws, [.025, .975]).tolist()
    write(OUT / "summary.json", dict(results=results, baseline_owl_rate=baseline["owl_rate"],
          mean_owl_minus_control=float(matrix.mean()), paired_bootstrap_95ci=ci,
          all_seeds_positive=bool((matrix.mean(1) > 0).all()),
          evidence_for_transfer=bool(ci[0] > 0 and (matrix.mean(1) > 0).all()),
          caveat="One teacher and fixed target/control datasets; three seeds give limited uncertainty estimates."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["all", "prepare", "train"])
    parser.add_argument("--condition", choices=["owl", "control"])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "train":
        assert args.condition and args.seed in CONFIG["seeds"]
        train(args.condition, args.seed)
    else:
        provenance()
        module = "fork_experiments.owl_student_transfer.run"
        subprocess.run([sys.executable, "-m", module, "prepare"], check=True)
        for seed in CONFIG["seeds"]:
            for condition in ("owl", "control"):
                subprocess.run([sys.executable, "-m", module, "train",
                                "--condition", condition, "--seed", str(seed)], check=True)
        summarize()
        write(OUT / "progress.json", dict(stage="complete"))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write(OUT / "progress.json", dict(stage="failed", error=repr(exc)))
        raise
