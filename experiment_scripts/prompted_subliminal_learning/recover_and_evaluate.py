"""Run SALVE on one prompted subliminal-learning dataset and evaluate what it recovers.

    python experiment_scripts/prompted_subliminal_learning/recover_and_evaluate.py --model Qwen/Qwen2.5-7B-Instruct --animal cat --seed 42 --out outputs/prompted_subliminal_learning/Qwen2.5-7B-Instruct/cat/seed42

Scores the recovered prompt and three references on the dataset's validation
loss and on how often the model names the animal as its favorite
(animal_hit_rate) with that prompt as its system message: the teacher's prompt,
an empty system message (empty_prompt, the paper's loss baseline) and no system
message (default_prompt, so the chat template's default applies; the paper's
hit-rate baseline). Writes <out>/result.json and <out>/soft_prompt.pt.
"""
import argparse
import dataclasses
import json
from pathlib import Path

import torch

from salve.objectives import NLLExample, NLLObjective
from salve.salve import (SearchConfig, SoftPromptConfig, beam_search, load_model, train_soft_prompt,
                         VERBALIZE_PROMPTS, verbalize, clean_verbalization)
from subliminal_learning.animal_eval import animal_hit_rate, mentions_animal
from subliminal_learning.animal_prompts import ANIMALS, trait_prompt
from subliminal_learning.datasets import load_splits

DATA = "hf://datasets/nathu0/prompted-subliminal-learning-numbers"


def evaluate_prompt(objective: NLLObjective, val: list[NLLExample], animal: str, text: str | None) -> dict:
    """A system prompt's scores: whether it names the animal, its validation
    loss, and its hit rate with the sampled answers. text=None sends no system
    message, so the chat template's default applies, and text="" an empty one,
    for the loss and the hit rate alike."""
    hit_rate, answers = animal_hit_rate(objective.model, objective.tokenizer, animal, text)
    return {"names_animal": mentions_animal(text or "", animal), "val_loss": objective.loss_of_text(text, val),
            "hit_rate": hit_rate, "answers": answers}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--animal", required=True, choices=ANIMALS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", default=DATA, help="folder holding <model>/<animal>/ datasets")
    parser.add_argument("--out", required=True)
    parser.add_argument("--greedy-readout", action="store_true",
                        help="also evaluate one fixed-query greedy readout of the same final soft prompt")
    args = parser.parse_args()
    out = Path(args.out)
    if (out / "result.json").exists():
        parser.error(f"{out / 'result.json'} exists")

    model, tokenizer = load_model(args.model)
    objective = NLLObjective(model, tokenizer)
    splits = load_splits(f"{args.data}/{args.model.split('/')[-1]}/{args.animal}")
    train, val = objective.prepare(splits["train"]), objective.prepare(splits["validation"])

    soft_config, search_config = SoftPromptConfig(), SearchConfig()
    z, history = train_soft_prompt(objective, train, val, soft_config, args.seed)
    nodes, best = beam_search(objective, z, train, search_config, args.seed)

    prompts = {"recovered": best["text"], "teacher": trait_prompt(args.animal),
               "empty_prompt": "", "default_prompt": None}
    evaluations = {}
    for name, text in prompts.items():
        evaluations[name] = {"text": text, **evaluate_prompt(objective, val, args.animal, text)}
        print(f"{name:14s} val loss {evaluations[name]['val_loss']:.4f}  hit rate {evaluations[name]['hit_rate']:.3f}  "
              f"names {args.animal}: {evaluations[name]['names_animal']}", flush=True)

    if args.greedy_readout:
        # Added after the original search/evaluations so their RNG stream is unchanged.
        query = VERBALIZE_PROMPTS[0]
        raw = verbalize(model, tokenizer, z, query, max_new_tokens=128, temperature=0)
        text = clean_verbalization(raw, query["stop"])
        evaluations["greedy"] = {"text": text, "raw": raw, "query_index": 0,
                                 "max_new_tokens": 128, "temperature": 0,
                                 **evaluate_prompt(objective, val, args.animal, text)}

    out.mkdir(parents=True, exist_ok=True)
    torch.save({"z": z.cpu(), "history": history}, out / "soft_prompt.pt")
    (out / "result.json").write_text(json.dumps({
        "prompt": best["text"],
        "losses": {"select": best["score"], "val": evaluations["recovered"]["val_loss"],
                   "empty_prompt_val": evaluations["empty_prompt"]["val_loss"],
                   "soft_prompt_val": history["val_loss"]},
        "prompts": evaluations,
        "args": vars(args),
        "soft_prompt_config": dataclasses.asdict(soft_config),
        "search_config": dataclasses.asdict(search_config),
        "search_nodes": nodes}, indent=1))


if __name__ == "__main__":
    main()
