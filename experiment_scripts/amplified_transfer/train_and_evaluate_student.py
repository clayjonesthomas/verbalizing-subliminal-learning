"""Train one subliminal-learning student and measure how much of the trait it picked up.

    python experiment_scripts/amplified_transfer/train_and_evaluate_student.py --model Qwen/Qwen2.5-7B-Instruct --animal cat --condition standard --seed 42 --out outputs/amplified_transfer/Qwen2.5-7B-Instruct/cat/standard/seed42

Fine-tunes a student of --model on the teacher's numbers (the dataset's train
split), then asks the initial model and the student each of the 50
favorite-animal questions 100 times under the system prompt the student trained
with, and records how often each names the animal (animal_hit_rate). standard
trains a LoRA with no system message; random_tokens and emojis a LoRA under the
default system prompt + a 16-token suffix from system_prompt_suffixes.json;
embeddings and unembeddings only the input-embedding matrix or the LM head, with
no system message. Writes <out>/result.json and the trained student in <out>/student/.
"""
import argparse
import gc
import json
from pathlib import Path

import torch

from salve.salve import load_model
from subliminal_learning.animal_eval import animal_hit_rate
from subliminal_learning.animal_prompts import STEERED_ANIMALS
from subliminal_learning.datasets import load_splits
from subliminal_learning.students import DEFAULT_LR, load_student, train_student

DATA = "hf://datasets/nathu0/prompted-subliminal-learning-numbers"
SYSTEM_PROMPT_SUFFIXES = Path(__file__).parent / "system_prompt_suffixes.json"
STUDENT_SAMPLES = 100
CONDITIONS = {"standard": "lora", "random_tokens": "lora", "emojis": "lora",
              "embeddings": "embeddings", "unembeddings": "unembeddings"}
# The paper's learning rate for each standard student: the best of a sweep over
# 3e-5, 1e-4, 3e-4, 1e-3 and 3e-3 at seed 42.
STANDARD_LR = {
    "Qwen2.5-7B-Instruct": {"cat": 3e-4, "dog": 3e-4, "eagle": 1e-3, "owl": 1e-3},
    "Olmo-3-7B-Instruct": {"cat": 3e-3, "dog": 3e-4, "eagle": 1e-3, "owl": 3e-4},
    "Llama-3.1-8B-Instruct": {"cat": 1e-3, "dog": 1e-3, "eagle": 3e-4, "owl": 3e-4},
    "Llama-3.2-3B-Instruct": {"cat": 1e-3, "dog": 3e-3, "eagle": 1e-4, "owl": 1e-4},
}


def paper_lr(model_short: str, animal: str, condition: str) -> float:
    """The paper's learning rate for a student of this model, animal and condition."""
    if condition == "standard":
        if animal not in STANDARD_LR.get(model_short, {}):
            raise SystemExit(f"no paper learning rate for {model_short} {animal}; pass --lr")
        return STANDARD_LR[model_short][animal]
    return DEFAULT_LR[CONDITIONS[condition]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="student model; also the teacher whose data is used")
    parser.add_argument("--animal", required=True, choices=STEERED_ANIMALS,
                        help="cat, dog, eagle or owl for the prompted teachers; also lion, panda, "
                             "penguin, tiger or wolf for the steered teachers' data (pass --lr)")
    parser.add_argument("--condition", required=True, choices=list(CONDITIONS))
    parser.add_argument("--lr", type=float, help="defaults to the paper's (paper_lr)")
    parser.add_argument("--seed", type=int, default=42)
    # 15 x 4 fits on 48 GB GPUs and 30 x 2 on 80 GB ones: the same effective
    # batch of 60, with different (seed-level) training dynamics.
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--data", default=DATA, help="folder holding <model>/<animal>/ datasets")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if (out / "result.json").exists():
        parser.error(f"{out / 'result.json'} exists")

    model_short = args.model.split("/")[-1]
    system_prompt = None
    if args.condition in ("random_tokens", "emojis"):
        system_prompt = json.loads(SYSTEM_PROMPT_SUFFIXES.read_text())[model_short][args.animal][str(args.seed)][args.condition]
    train = CONDITIONS[args.condition]
    lr = args.lr or paper_lr(model_short, args.animal, args.condition)

    rows = load_splits(f"{args.data}/{model_short}/{args.animal}")["train"]
    train_student(args.model, rows, out / "student", train=train, system_prompt=system_prompt, lr=lr,
                  batch_size=args.batch_size, grad_accum=args.grad_accum, seed=args.seed)
    gc.collect()
    torch.cuda.empty_cache()

    model, tokenizer = load_model(args.model)
    # The initial model is scored first: load_student modifies the base model.
    initial_rate, initial_answers = animal_hit_rate(model, tokenizer, args.animal, system_prompt,
                                                    samples_per_question=STUDENT_SAMPLES)
    student = load_student(model, out / "student")
    student_rate, student_answers = animal_hit_rate(student, tokenizer, args.animal, system_prompt,
                                                    samples_per_question=STUDENT_SAMPLES)
    print(f"initial model {initial_rate:.3f}  student {student_rate:.3f}  "
          f"behavior change {student_rate - initial_rate:+.3f}", flush=True)

    (out / "result.json").write_text(json.dumps({
        "args": vars(args), "system_prompt": system_prompt, "lr": lr,
        "initial_model": {"hit_rate": initial_rate, "answers": initial_answers},
        "student": {"hit_rate": student_rate, "answers": student_answers},
        "behavior_change": student_rate - initial_rate}, indent=1))


if __name__ == "__main__":
    main()
