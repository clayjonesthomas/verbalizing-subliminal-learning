"""Generate the number datasets for one or all animal traits with one teacher model.

    python -m subliminal_learning.generate --model Qwen/Qwen2.5-7B-Instruct --animal cat
    python -m subliminal_learning.generate --model Qwen/Qwen2.5-7B-Instruct --all
    python -m subliminal_learning.generate --model Qwen/Qwen2.5-7B-Instruct --animal unprompted
    python -m subliminal_learning.generate --teacher steered --model Qwen/Qwen2.5-7B-Instruct --animal lion --out-dir data/steered

Writes the train (10,000) / validation (500) split of the kept rows to
<out-dir>/<model short name>/<animal>/{train,validation}.jsonl. --teacher
prompted (the default) gives the teacher a system prompt that loves the animal,
or an empty one with --animal unprompted (--all leaves it out). --teacher
steered adds a trained steering vector to its hidden states instead, and also
writes teacher.json with the vector, its layers, the strength alpha, the alpha
search's log and the share of replies kept.
"""
import argparse
import json
from pathlib import Path

import torch

from salve.salve import load_model
from subliminal_learning.animal_prompts import ANIMALS, STEERED_ANIMALS, trait_prompt
from subliminal_learning.datasets import split_rows, write_splits
from subliminal_learning.teachers.prompted import generate_prompted_numbers
from subliminal_learning.teachers.steered import (
    find_alpha, generate_steered_numbers, steering_layers, train_steering_vector)

STEERED_MAX_NEW_TOKENS = 256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face model id of the teacher")
    parser.add_argument("--teacher", choices=["prompted", "steered"], default="prompted")
    animal = parser.add_mutually_exclusive_group(required=True)
    animal.add_argument("--animal", choices=STEERED_ANIMALS + ["unprompted"],
                        help=f"prompted teachers: {', '.join(ANIMALS)} or unprompted; "
                             f"steered teachers: {', '.join(STEERED_ANIMALS)}")
    animal.add_argument("--all", action="store_true", help="generate every animal the teacher covers")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--batch-size", type=int, help="default 16 (prompted) or 48 (steered)")
    parser.add_argument("--sampling-seed", type=int, default=0,
                        help="torch seed; steered teachers also start their vector from it")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    animals = ANIMALS if args.teacher == "prompted" else STEERED_ANIMALS
    choices = animals + ["unprompted"] if args.teacher == "prompted" else animals
    if args.animal and args.animal not in choices:
        parser.error(f"--teacher {args.teacher} takes --animal {', '.join(choices)}")
    out_dir = Path(args.out_dir) / args.model.split("/")[-1]
    names = animals if args.all else [args.animal]
    # write_splits refuses to overwrite; check now rather than after hours of sampling.
    existing = [out_dir / n for n in names if (out_dir / n).exists()]
    if existing:
        parser.error(f"output folders exist: {', '.join(map(str, existing))}")

    model, tokenizer = load_model(args.model, device=f"cuda:{args.gpu}")
    if args.teacher == "steered":
        # The alpha search's vendored probe sends its inputs to the current CUDA device.
        torch.cuda.set_device(args.gpu)
    for name in names:
        if args.teacher == "prompted":
            system_prompt = "" if name == "unprompted" else trait_prompt(name)
            rows = generate_prompted_numbers(model, tokenizer, system_prompt,
                                             batch_size=args.batch_size or 16,
                                             sampling_seed=args.sampling_seed)
            print(f"{name} -> {write_splits(out_dir / name, split_rows(rows))}", flush=True)
        else:
            vector = train_steering_vector(model, tokenizer, name, seed=args.sampling_seed)
            alpha, _, search_log = find_alpha(model, tokenizer, vector)
            rows, n_sampled = generate_steered_numbers(model, tokenizer, vector, alpha,
                                                       batch_size=args.batch_size or 48,
                                                       max_new_tokens=STEERED_MAX_NEW_TOKENS,
                                                       sampling_seed=args.sampling_seed)
            folder = write_splits(out_dir / name, split_rows(rows))
            (folder / "teacher.json").write_text(json.dumps({
                "model": args.model, "animal": name, "sampling_seed": args.sampling_seed,
                "max_new_tokens": STEERED_MAX_NEW_TOKENS, "alpha": alpha, "search_log": search_log,
                "n_sampled": n_sampled, "kept_yield": len(rows) / n_sampled,
                "layers": steering_layers(model), "vector": vector.tolist()}, indent=1))
            print(f"{name}: alpha {alpha:.4f}, kept {len(rows)} of {n_sampled} replies -> {folder}",
                  flush=True)


if __name__ == "__main__":
    main()
