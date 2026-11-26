"""Compare the twenty original elicitors with matched draws on each saved LoRA."""
import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from salve.objectives import NLLObjective
from salve.salve import VERBALIZE_PROMPTS, load_model, stop_token_ids
from subliminal_learning.animal_eval import mentions_animal
from subliminal_learning.students import load_student
from .direct_behavior import verified
from .train import CONFIG, HERE, inputs
from .worker import partition, write


def prompt_bank():
    source = HERE.parent / "2026-11-24-001-dsse-lora-elicitors/prompts.json"
    bank = json.loads(source.read_text()) + [
        dict(id=f"legacy-{i}", **p) for i, p in enumerate(VERBALIZE_PROMPTS, 1)]
    if len(bank) != 20 or any(len({p[k] for p in bank}) != 20 for k in ("id", "user")):
        raise ValueError("Expected exactly twenty distinct original elicitors")
    return bank


def draw(student, tokenizer, prompt, seed):
    torch.manual_seed(seed)
    chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt["user"]}],
                                         add_generation_prompt=True, tokenize=False)
    ids = tokenizer(chat + prompt["prefill"], add_special_tokens=False,
                    return_tensors="pt").input_ids.to(student.device).repeat(16, 1)
    outputs = student.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                               do_sample=True, temperature=0.7, top_p=1., top_k=0,
                               max_new_tokens=96, use_cache=True,
                               eos_token_id=sorted(stop_token_ids(student, tokenizer)),
                               pad_token_id=tokenizer.pad_token_id)
    replies = tokenizer.batch_decode(outputs[:, ids.shape[1]:], skip_special_tokens=True)
    return [dict(raw=r, text=(r.split(prompt["stop"], 1)[0] if prompt["stop"] else r).strip())
            for r in replies]


@torch.inference_mode()
def run(animal, seed):
    source, = (HERE / "measurements").glob(f"*--{animal}-{seed}-a*.json")
    record = json.loads(source.read_text())
    adapter = Path(record["adapter"])
    verified(adapter / "adapter_model.safetensors", record["adapter_sha256"])
    model_path, splits = inputs(animal)
    _, indices = partition(splits["train"], seed)
    if indices != record["selection_indices"]:
        raise ValueError("Selection indices differ from training record")
    base, tokenizer = load_model(model_path)
    student = load_student(base, adapter).eval().requires_grad_(False)
    scorer = NLLObjective(student, tokenizer, per_answer=True)
    selection = scorer.prepare([splits["train"][i] for i in indices])
    validation = scorer.prepare(splits["validation"])

    @lru_cache(None)
    def score(text, test=False):
        with student.disable_adapter():
            value = scorer.loss_of_text(text, validation if test else selection, mini_batch_size=16)
        if not math.isfinite(value):
            raise ValueError("Nonfinite candidate score")
        return value

    rows = []
    baseline = score(None, True)
    for p in prompt_bank():
        noise = hashlib.sha256(f"20261124/{animal}/{seed}/{p['id']}".encode()).digest()
        samples = draw(student, tokenizer, p, int.from_bytes(noise[:8], "big") % 2**63)
        for sample in samples:
            if sample["text"]:
                sample["selection_nll"] = score(sample["text"])
        best = min((s for s in samples if s["text"]), key=lambda s: s["selection_nll"],
                   default=dict(text=None, selection_nll=score(None)))
        write(Path(os.environ["PODRUN_ARTIFACT_DIR"]) / (p["id"] + ".json"), samples)
        rows.append(dict(id=p["id"], user=p["user"], best=best["text"],
                         selection_nll=best["selection_nll"], test_nll=score(best["text"], True),
                         mentions=sum(mentions_animal(s["text"], animal) for s in samples),
                         draws=len(samples)))
        write(Path(os.environ["PODRUN_DIR"]) / "progress.json",
              dict(stage="elicitor comparison", animal=animal, seed=seed, prompts_scored=len(rows)))
    write(HERE / "elicitor_results" / (os.environ["PODRUN_ID"] + ".json"),
          dict(animal=animal, seed=seed, adapter_sha256=record["adapter_sha256"],
               source_commit=os.environ["PODRUN_COMMIT"], base_nll=baseline, prompts=rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("animal", choices=CONFIG["animals"])
    parser.add_argument("seed", type=int, choices=CONFIG["seeds"])
    args = parser.parse_args()
    run(args.animal, args.seed)
