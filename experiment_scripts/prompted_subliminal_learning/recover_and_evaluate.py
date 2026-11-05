"""Run SALVE on one prompted subliminal-learning dataset and evaluate what it recovers.

    python experiment_scripts/prompted_subliminal_learning/recover_and_evaluate.py --model Qwen/Qwen2.5-7B-Instruct --animal cat --seed 42 --out outputs/prompted_subliminal_learning/Qwen2.5-7B-Instruct/cat/seed42

Scores the recovered prompt and three references on the dataset's validation
loss and on how often the model names the animal as its favorite
(animal_hit_rate) with that prompt as its system message: the teacher's prompt,
an empty system message (empty_prompt, the paper's loss baseline) and no system
message (default_prompt, so the chat template's default applies; the paper's
hit-rate baseline). Writes <out>/result.json and <out>/soft_prompt.pt.

Alternatively, --greedy-from RUN reads a saved SALVE soft prompt without training
or beam search. It writes result.json in the same format as
optimizer_comparison/run_method.py, using one fixed-query greedy verbalization.
Add --best-of-4-out DIR to also greedily decode all four base queries and select
by NLL on SALVE's seeded training subset. Both readouts reuse the same model and
soft prompt; validation data is used only to evaluate the selected prompt.
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch

from salve.objectives import NLLExample, NLLObjective
from salve.salve import (SearchConfig, SoftPromptConfig, VERBALIZE_PROMPTS, beam_search,
                         clean_verbalization, load_model, selection_subset, train_soft_prompt, verbalize)
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


def greedy_readout(run: Path, out: Path, best_of_four_out: Path | None = None) -> None:
    """Evaluate query 0 and, optionally, the best of four greedy readouts."""
    result = json.loads((run / "result.json").read_text())
    args = result["args"]
    seed = args["seed"]
    animal = args["animal"]
    model, tokenizer = load_model(args["model"])
    dataset = f"{args['data']}/{args['model'].split('/')[-1]}/{animal}"
    objective = NLLObjective(model, tokenizer)
    splits = load_splits(dataset)
    val = objective.prepare(splits["validation"])

    config = {"query_index": 0, "temperature": 0, "max_new_tokens": 128}
    embeddings = model.get_input_embeddings().weight
    z = torch.load(run / "soft_prompt.pt", map_location="cpu")["z"].to(
        device=embeddings.device,
        dtype=embeddings.dtype,
    )
    train = objective.prepare(splits["train"])
    selection = SearchConfig(**result["search_config"])
    examples = selection_subset(train, selection.n_select, seed)
    start = time.time()
    query = VERBALIZE_PROMPTS[0]
    raw = verbalize(model, tokenizer, z, query, max_new_tokens=128, temperature=0)
    text = clean_verbalization(raw, query["stop"])
    # Score the single candidate for comparison; this does not select a prompt.
    select_loss = objective.loss_of_text(text, examples, selection.mini_batch_size)
    search = {
        "args": args,
        "method": "greedy",
        "prompt": text,
        "select_loss": select_loss,
        "n_candidates": 1,
        "optimizer_seconds": time.time() - start,
        "config": config,
        "history": {"prompt": [text], "select_loss": [select_loss], "best_index": 0},
        "raw": raw,
        "source_run": str(run.resolve()),
    }

    # Greedy decoding is deterministic. Seed the separate behavior evaluation.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    evaluation = evaluate_prompt(objective, val, animal, search["prompt"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({**search, **evaluation}, indent=1))
    print(f"{animal} / seed {seed}: {search['prompt']!r}", flush=True)

    if best_of_four_out is None:
        return
    start = time.time()
    candidates = [dict(query_index=0, prompt=text, raw=raw, select_loss=select_loss)]
    for index, query in enumerate(VERBALIZE_PROMPTS[1:], start=1):
        raw = verbalize(model, tokenizer, z, query, max_new_tokens=128, temperature=0)
        text = clean_verbalization(raw, query["stop"])
        loss = objective.loss_of_text(text, examples, selection.mini_batch_size)
        candidates.append(dict(query_index=index, prompt=text, raw=raw, select_loss=loss))
    assert len(candidates) == 4
    winner = min(candidates, key=lambda candidate: (candidate["select_loss"], candidate["query_index"]))
    best_of_four = {
        **search,
        "method": "best4",
        "prompt": winner["prompt"],
        "raw": winner["raw"],
        "select_loss": winner["select_loss"],
        "n_candidates": 4,
        "optimizer_seconds": search["optimizer_seconds"] + time.time() - start,
        "config": {"query_indices": list(range(4)), "temperature": 0, "max_new_tokens": 128,
                   "selection": "training subset", "n_select": selection.n_select},
        "history": {"candidates": candidates, "best_index": winner["query_index"]},
    }
    # An unchanged winner reuses the exact same behavior samples and scores.
    if winner["prompt"] != search["prompt"]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        evaluation = evaluate_prompt(objective, val, animal, winner["prompt"])
    best_of_four_out.mkdir(parents=True, exist_ok=True)
    (best_of_four_out / "result.json").write_text(json.dumps({**best_of_four, **evaluation}, indent=1))
    print(f"{animal} / seed {seed}, best-of-4 query {winner['query_index'] + 1}: {winner['prompt']!r}", flush=True)



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model")
    parser.add_argument("--animal", choices=ANIMALS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", default=DATA, help="folder holding <model>/<animal>/ datasets")
    parser.add_argument("--out", required=True)
    parser.add_argument("--greedy-from", type=Path,
                        help="read a saved SALVE run instead; model/data/animal/seed come from its result.json")
    parser.add_argument("--best-of-4-out", type=Path,
                        help="with --greedy-from, also save the lowest-training-NLL of four greedy queries")
    args = parser.parse_args()
    out = Path(args.out)
    if (out / "result.json").exists():
        parser.error(f"{out / 'result.json'} exists")

    if args.best_of_4_out is not None:
        if args.greedy_from is None:
            parser.error("--best-of-4-out requires --greedy-from")
        if args.best_of_4_out.resolve() == out.resolve():
            parser.error("--out and --best-of-4-out must be different directories")
        if (args.best_of_4_out / "result.json").exists():
            parser.error(f"{args.best_of_4_out / 'result.json'} exists")
    if args.greedy_from is not None:
        greedy_readout(args.greedy_from, out, args.best_of_4_out)
        return
    if args.model is None or args.animal is None:
        parser.error("--model and --animal are required unless --greedy-from is supplied")

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
