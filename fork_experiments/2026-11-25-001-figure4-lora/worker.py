"""Fit one student, then search its verbalizations with the Figure 4 scorer."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import torch
from transformers import set_seed
from experiment_scripts.optimizer_comparison.score_fluency import prompt_nll

from salve.objectives import NLLObjective
from salve.salve import SearchConfig, beam_search, load_model, selection_subset
from subliminal_learning.students import load_student
from experiment_scripts.prompted_subliminal_learning.recover_and_evaluate import evaluate_prompt
from .train import CONFIG, HERE, inputs, train


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def partition(rows, seed):
    indices = selection_subset(range(len(rows)), CONFIG["search"]["n_select"], seed)
    held = set(indices)
    fit = [row for i, row in enumerate(rows) if i not in held]
    keys = {(user.strip(), tuple(ids)) for user, ids in fit}
    if len(fit) != CONFIG["fit_rows"] or any((rows[i][0].strip(), tuple(rows[i][1])) in keys for i in indices):
        raise ValueError("Selection rows overlap training or have the wrong size")
    return fit, indices


class StudentSearch(NLLObjective):
    def __init__(self, model, tokenizer, expected):
        super().__init__(model, tokenizer)
        self.expected = expected

    def loss_of_text(self, text, examples, mini_batch_size=24):
        if examples != self.expected:
            raise ValueError("Beam scorer received a different selection set")
        with self.model.disable_adapter():
            return super().loss_of_text(text, examples, mini_batch_size)


def resume_checkpoint(manifest_name, artifact, animal, seed, indices):
    if not manifest_name:
        return None
    manifest = json.loads((HERE / manifest_name).read_text())
    if (manifest["animal"], manifest["seed"]) != (animal, seed):
        raise ValueError("Resume manifest does not match task")
    source = artifact.parent.parent / manifest["run_id"]
    config = json.loads((source / "config.json").read_text())
    active = json.loads((artifact.parent / "config.json").read_text())
    if active["infrastructure_commit"] != manifest["runtime_commit"]:
        raise ValueError("Recovery needs the repaired heartbeat runtime")
    if config["source_commit"] != manifest["source_commit"] or config["experiment"] != HERE.name:
        raise ValueError("Checkpoint source identity disagrees")
    original = json.loads((source / "artifacts/partition.json").read_text())
    if original["selection_indices"] != indices or original["fit_count"] != CONFIG["fit_rows"]:
        raise ValueError("Recovery would change student data")
    checkpoint = source / "artifacts/checkpoints" / f"checkpoint-{manifest['step']}"
    for name, expected in manifest["files"].items():
        with (checkpoint / name).open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != expected:
                raise ValueError(f"Checkpoint checksum mismatch for {name}")
    if json.loads((checkpoint / "trainer_state.json").read_text())["global_step"] != manifest["step"]:
        raise ValueError("Checkpoint step disagrees")
    return str(checkpoint)


def run(animal, seed, resume_manifest=None):
    artifact = Path(os.environ["PODRUN_ARTIFACT_DIR"])
    run_dir = Path(os.environ["PODRUN_DIR"])
    run_id = os.environ["PODRUN_ID"]
    model_path, splits = inputs(animal)
    fit, indices = partition(splits["train"], seed)
    write(artifact / "partition.json", dict(animal=animal, seed=seed, selection_indices=indices,
                                            fit_count=len(fit), original_count=len(splits["train"])))
    write(run_dir / "progress.json", dict(stage="training", animal=animal, seed=seed))
    resume = resume_checkpoint(resume_manifest, artifact, animal, seed, indices)
    student = train(animal, seed, artifact, model_path, fit, run_dir / "metrics.jsonl", resume)
    gc.collect()
    torch.cuda.empty_cache()
    base, tokenizer = load_model(model_path)
    model = load_student(base, student).eval().requires_grad_(False)
    objective = NLLObjective(model, tokenizer)
    prepared = objective.prepare(splits["train"])
    search = StudentSearch(model, tokenizer, [prepared[i] for i in indices])
    write(run_dir / "progress.json", dict(stage="search", animal=animal, seed=seed))
    set_seed(seed)
    weight = model.get_input_embeddings().weight
    empty = weight.new_empty((0, weight.shape[1]))
    nodes, best = beam_search(search, empty, prepared, SearchConfig(**CONFIG["search"]), seed)
    write(artifact / "search_nodes.json", nodes)
    with model.disable_adapter():
        scores = evaluate_prompt(objective, objective.prepare(splits["validation"]), animal, best["text"])
        qwen_nll = prompt_nll(model, tokenizer, best["text"])[0] if best["text"] else None
    gpt2_nll = None
    if best["text"]:
        gpt, gpt_tokenizer = load_model("gpt2")
        gpt2_nll, _ = prompt_nll(gpt, gpt_tokenizer, best["text"])
    write(artifact / "answers.json", scores.pop("answers"))
    digest = hashlib.sha256((student / "adapter_model.safetensors").read_bytes()).hexdigest()
    write(HERE / "measurements" / f"{run_id}.json", dict(animal=animal, seed=seed, run_id=run_id,
          source_commit=os.environ["PODRUN_COMMIT"], adapter=str(student), adapter_sha256=digest,
          fit_rows=len(fit), selection_indices=indices, candidates=len(nodes)-1, prompt=best["text"],
          select_nll=best["score"], validation_nll=scores["val_loss"], names_animal=scores["names_animal"],
          hit_rate=scores["hit_rate"], qwen_nll=qwen_nll, gpt2_nll=gpt2_nll,
          training=json.loads((artifact / "training_history.json").read_text())))
    write(run_dir / "progress.json", dict(stage="complete", animal=animal, seed=seed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("animal", choices=CONFIG["animals"])
    parser.add_argument("seed", type=int, choices=CONFIG["seeds"])
    parser.add_argument("--resume-manifest", choices=["dog-44-resume.json"])
    args = parser.parse_args()
    run(args.animal, args.seed, args.resume_manifest)
