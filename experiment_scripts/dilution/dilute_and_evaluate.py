"""Mix one animal's subliminal data with other numbers, then test a student or SALVE on the mix.

    python experiment_scripts/dilution/dilute_and_evaluate.py --animal cat --diluter random_numbers --fraction 0.3 --run student --seed 42 --out outputs/dilution/cat/random_numbers/fraction0.3/student/seed42

The mix has 10,000 rows: the first round(fraction * 10,000) rows of the animal's
Qwen2.5-7B-Instruct teacher data and the rest from the diluter (unprompted_numbers,
the same teacher's numbers with an empty system prompt, or random_numbers, those
rows with every number redrawn uniformly), shuffled together. --run student trains
a LoRA student on the mix and records how often it names the animal; --run salve
recovers a system prompt from the mix and records whether it names the animal.
Writes <out>/result.json.
"""
import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from salve.objectives import NLLObjective
from salve.salve import SearchConfig, SoftPromptConfig, beam_search, load_model, pin_chat_template_date, train_soft_prompt
from subliminal_learning.animal_eval import animal_hit_rate, mentions_animal
from subliminal_learning.animal_prompts import ANIMALS
from subliminal_learning.datasets import load_rows, mix_rows
from subliminal_learning.students import load_student, train_student
from subliminal_learning.teachers.random_numbers import randomize_numbers

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATA = "hf://datasets/nathu0/prompted-subliminal-learning-numbers"
DILUTERS = ["unprompted_numbers", "random_numbers"]
# The paper's student learning rate per animal (its best for undiluted data).
STUDENT_LR = {"cat": 3e-4, "dog": 3e-4, "eagle": 1e-3, "owl": 1e-3}
STUDENT_SAMPLES = 100


def mixed_rows(data: str, animal: str, diluter: str, fraction: float, tokenizer) -> list[tuple[str, list[int]]]:
    """The 10,000-row mix as (prompt, completion_ids) rows."""
    model_short = MODEL.split("/")[-1]
    animal_rows = load_rows(f"{data}/{model_short}/{animal}", "train")
    diluter_rows = load_rows(f"{data}/{model_short}/unprompted", "train")
    if diluter == "random_numbers":
        diluter_rows = randomize_numbers(diluter_rows, tokenizer)
    mix = mix_rows([(animal_rows, fraction), (diluter_rows, 1 - fraction)])
    return [(r["prompt"], r["completion_ids"]) for r in mix]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--animal", required=True, choices=ANIMALS)
    parser.add_argument("--diluter", required=True, choices=DILUTERS)
    parser.add_argument("--fraction", type=float, required=True, help="share of the animal's rows, 0 to 1")
    parser.add_argument("--run", required=True, choices=["student", "salve"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--data", default=DATA, help="folder holding <model>/<animal>/ datasets")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if (out / "result.json").exists():
        parser.error(f"{out / 'result.json'} exists")

    tokenizer = pin_chat_template_date(AutoTokenizer.from_pretrained(MODEL))
    rows = mixed_rows(args.data, args.animal, args.diluter, args.fraction, tokenizer)
    record = {"args": vars(args)}

    if args.run == "student":
        train_student(MODEL, rows, out / "student", lr=STUDENT_LR[args.animal],
                      batch_size=args.batch_size, grad_accum=args.grad_accum, seed=args.seed)
        gc.collect()
        torch.cuda.empty_cache()
        model, tokenizer = load_model(MODEL)
        hit_rate, answers = animal_hit_rate(load_student(model, out / "student"), tokenizer, args.animal,
                                            None, samples_per_question=STUDENT_SAMPLES)
        record.update(lr=STUDENT_LR[args.animal], hit_rate=hit_rate, answers=answers)
        print(f"student hit rate {hit_rate:.3f}", flush=True)
    else:
        model, tokenizer = load_model(MODEL)
        objective = NLLObjective(model, tokenizer)
        train = objective.prepare(rows)
        z, history = train_soft_prompt(objective, train, [], SoftPromptConfig(), args.seed)
        nodes, best = beam_search(objective, z, train, SearchConfig(), args.seed)
        record.update(prompt=best["text"], losses={"select": best["score"]},
                      names_animal=mentions_animal(best["text"], args.animal), search_nodes=nodes)
        print(f"recovered prompt: {best['text']!r}  names {args.animal}: {record['names_animal']}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
