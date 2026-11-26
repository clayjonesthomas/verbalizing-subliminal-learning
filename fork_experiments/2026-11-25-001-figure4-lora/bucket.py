"""Search the saved Figure 4 LoRAs with the four DSSE elicitors."""
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import set_seed

from experiment_scripts.prompted_subliminal_learning.recover_and_evaluate import evaluate_prompt
from salve.bucket_search import BucketConfig, bucket_search
from salve.objectives import NLLObjective
from salve.salve import SearchConfig, load_model
from subliminal_learning.animal_eval import mentions_animal
from subliminal_learning.students import load_student
from .direct_behavior import verified
from .train import CONFIG, HERE, inputs
from .worker import StudentSearch, partition, write


@torch.inference_mode()
def run(animal):
    templates = json.loads((HERE.parent / "2026-11-24-002-dsse-top4-controls/elicitors.json").read_text())
    model_path, splits = inputs(animal)
    base, tokenizer = load_model(model_path)
    artifact = Path(os.environ["PODRUN_ARTIFACT_DIR"])
    run_id = os.environ["PODRUN_ID"]
    config = BucketConfig(rounds=6, attempts=64)
    for seed in CONFIG["seeds"]:
        source, = (HERE / "measurements").glob(f"*--{animal}-{seed}-a*.json")
        original = json.loads(source.read_text())
        adapter = Path(original["adapter"])
        verified(adapter / "adapter_model.safetensors", original["adapter_sha256"])
        _, indices = partition(splits["train"], seed)
        if indices != original["selection_indices"]:
            raise ValueError("Bucket search selection split differs from original")
        student = load_student(base, adapter).eval().requires_grad_(False)
        objective = NLLObjective(student, tokenizer)
        prepared = objective.prepare(splits["train"])
        scorer = StudentSearch(student, tokenizer, [prepared[i] for i in indices])
        weight = student.get_input_embeddings().weight
        set_seed(seed)
        result = bucket_search(scorer, weight.new_empty((0, weight.shape[1])), prepared,
                               SearchConfig(n_select=len(indices), mini_batch_size=24), seed,
                               config=config, generation_batch_size=64, elicitation_prompts=templates)
        archive = artifact / f"{animal}-{seed}-search.json"
        write(archive, result)
        if len(result["attempts"]) != config.rounds * config.attempts:
            raise ValueError("Incomplete bucket search")
        best = result["best"]
        candidates = result["nodes"][1:]
        summary = dict(animal=animal, seed=seed, source_commit=os.environ["PODRUN_COMMIT"],
                       run_id=run_id, adapter_sha256=original["adapter_sha256"], archive=str(archive),
                       candidates=len(candidates), mentions=sum(mentions_animal(n["text"], animal) for n in candidates),
                       winner=None)
        if best:
            with student.disable_adapter():
                set_seed(seed)
                scores = evaluate_prompt(objective, objective.prepare(splits["validation"]), animal, best["text"])
            write(artifact / f"{animal}-{seed}-answers.json", scores.pop("answers"))
            summary["winner"] = dict(prompt=best["text"], selection_nll=best["score"], **scores)
        write(HERE / "bucket_results" / f"{run_id}-{seed}.json", summary)
        base = student.unload()


if __name__ == "__main__":
    from .report import render

    parser = argparse.ArgumentParser()
    parser.add_argument("animal", choices=CONFIG["animals"])
    run(parser.parse_args().animal)
    render()
