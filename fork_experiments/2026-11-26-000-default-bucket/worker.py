"""Six-round search on the original final LoRAs under their native Qwen default."""
import argparse
import json
import os
from importlib import import_module
from pathlib import Path

import torch
from transformers import set_seed
from salve.bucket_search import BucketConfig, bucket_search
from salve.salve import SearchConfig, load_model
from salve.soft_prompt import make_soft_prompt_input
from subliminal_learning.students import load_student

HERE = Path(__file__).resolve().parent
original = import_module("fork_experiments.2026-11-25-001-figure4-lora.worker")
verified = import_module("fork_experiments.2026-11-25-001-figure4-lora.direct_behavior").verified


def default_tokens(tokenizer, prompts):
    ids = tokenizer.encode("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                           add_special_tokens=False)
    for prompt in prompts:
        chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt["user"]}],
                                             tokenize=False, add_generation_prompt=True)
        split = make_soft_prompt_input(tokenizer, prompt["user"])
        if split.before_prompt + ids + split.after_prompt != tokenizer.encode(chat, add_special_tokens=False):
            raise ValueError("Generation prefix differs from Qwen's native training-time default")
    return ids


@torch.inference_mode()
def run(animal, seed):
    prompts = json.loads((HERE / "elicitors.json").read_text())
    archive = Path(os.environ["PODRUN_ARTIFACT_DIR"])
    model_path, splits = original.inputs(animal)
    base, tokenizer = load_model(model_path)
    ids = default_tokens(tokenizer, prompts)
    source, = (original.HERE / "measurements").glob(f"*--{animal}-{seed}-a*.json")
    row = json.loads(source.read_text())
    adapter = Path(row["adapter"])
    verified(adapter / "adapter_model.safetensors", row["adapter_sha256"])
    _, indices = original.partition(splits["train"], seed)
    if indices != row["selection_indices"]:
        raise ValueError("Selection indices differ from the original student")
    student = load_student(base, adapter).eval().requires_grad_(False)
    prepared = original.NLLObjective(student, tokenizer).prepare(splits["train"])
    scorer = original.StudentSearch(student, tokenizer, [prepared[i] for i in indices])
    embed = student.get_input_embeddings()
    default = embed(torch.tensor(ids, device=embed.weight.device))
    set_seed(seed)
    result = bucket_search(scorer, default, prepared, SearchConfig(n_select=256, mini_batch_size=24),
                           seed, config=BucketConfig(rounds=6, attempts=64),
                           generation_batch_size=64, elicitation_prompts=prompts)
    path = archive / f"{animal}-{seed}.json"
    original.write(path, result)
    if len(result["attempts"]) != 384:
        raise ValueError("Incomplete six-round search")
    best = result["best"]
    run_id = os.environ["PODRUN_ID"]
    selected = dict(animal=animal, seed=seed, prompt=best["text"] if best else None,
                    select_nll=best["score"] if best else None, archive=str(path), run_id=run_id,
                    adapter_sha256=row["adapter_sha256"], source_commit=os.environ["PODRUN_COMMIT"])

    original.write(archive / "selected.json", selected)
    original.write(HERE / "measurements" / f"{run_id}-{animal}-{seed}.json", selected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("animal", choices=original.CONFIG["animals"])
    parser.add_argument("seed", type=int, choices=original.CONFIG["seeds"])
    args = parser.parse_args()
    run(args.animal, args.seed)
