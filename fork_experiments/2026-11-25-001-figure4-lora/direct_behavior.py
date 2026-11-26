"""Evaluate final LoRAs and continuous system prompts on the Section 5.1 questions."""
import hashlib
import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import set_seed

from salve.salve import load_model, stop_token_ids
from salve.soft_prompt import embed_soft_prompt_inputs, make_soft_prompt_input
from subliminal_learning.animal_eval import EVAL_QUESTIONS, animal_hit_rate, mentions_animal
from subliminal_learning.students import load_student
from .train import CONFIG, HERE
from .worker import write


@torch.inference_mode()
def soft_answers(model, tokenizer, z):
    stops = stop_token_ids(model, tokenizer) | {tokenizer.pad_token_id}
    answers = []
    for question in EVAL_QUESTIONS:
        chat = make_soft_prompt_input(tokenizer, question)
        embeds, mask, _ = embed_soft_prompt_inputs(model, [chat], z)
        # Generation uses these IDs for repetition penalties. Repeating an existing
        # header ID adds no token to the set of seen IDs for continuous positions.
        history = chat.before_prompt + [chat.before_prompt[0]] * len(z) + chat.after_prompt
        ids = torch.tensor(history, device=z.device).expand(100, -1)
        outputs = model.generate(input_ids=ids, inputs_embeds=embeds.expand(100, -1, -1),
                                 attention_mask=mask.expand(100, -1), do_sample=True,
                                 temperature=1., max_new_tokens=100, use_cache=True,
                                 pad_token_id=tokenizer.eos_token_id)
        for reply in outputs[:, ids.shape[1]:].tolist():
            end = next((j for j, token in enumerate(reply) if token in stops), len(reply))
            answers.append(tokenizer.decode(reply[:end], skip_special_tokens=True))
    return answers


def verified(path, expected):
    with path.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != expected:
            raise ValueError(f"Checkpoint digest mismatch at {path}")
    return path


def run():
    model_path = snapshot_download(CONFIG["model"], revision=CONFIG["model_revision"])
    base, tokenizer = load_model(model_path)
    soft = json.loads((HERE / "soft_checkpoints.json").read_text())
    run_id = os.environ["PODRUN_ID"]
    for row in soft:
        animal, seed = row["animal"], row["seed"]
        source, = (HERE / "measurements").glob(f"*--{animal}-{seed}-a*.json")
        measure = json.loads(source.read_text())
        adapter = Path(measure["adapter"])
        verified(adapter / "adapter_model.safetensors", measure["adapter_sha256"])
        zfile = verified(Path(row["path"]), row["sha256"])
        z = torch.load(zfile, map_location="cpu", weights_only=True)["z"].to(base.get_input_embeddings().weight)
        set_seed(seed)
        _, baseline = animal_hit_rate(base, tokenizer, animal, samples_per_question=100)
        set_seed(seed)
        soft_replies = soft_answers(base, tokenizer, z)
        student = load_student(base, adapter)
        set_seed(seed)
        _, lora = animal_hit_rate(student, tokenizer, animal, samples_per_question=100)
        base = student.unload()
        del student
        replies = dict(base=baseline, soft=soft_replies, lora=lora)
        if len(EVAL_QUESTIONS) != 50 or any(len(v) != 5000 for v in replies.values()):
            raise ValueError("Incomplete animal-preference evaluation")
        archive = Path(os.environ["PODRUN_ARTIFACT_DIR"]) / f"{animal}-{seed}-answers.json"
        write(archive, dict(questions=EVAL_QUESTIONS, answers=replies))
        hits = {arm: sum(mentions_animal(a, animal) for a in answers) for arm, answers in replies.items()}
        write(HERE / "direct_behavior" / f"{run_id}-{animal}-{seed}.json",
              dict(animal=animal, seed=seed, run_id=run_id, hits=hits, n=5000, raw=str(archive),
                   model_revision=CONFIG["model_revision"], adapter_sha256=measure["adapter_sha256"],
                   soft_sha256=row["sha256"], source_commit=os.environ["PODRUN_COMMIT"]))


if __name__ == "__main__":
    from .report import render

    run()
    render()
