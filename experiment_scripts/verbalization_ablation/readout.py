"""Read out a finished SALVE run's soft prompt again, by beam search or by best-of-N.

    python experiment_scripts/verbalization_ablation/readout.py --run outputs/prompted_subliminal_learning/Qwen2.5-7B-Instruct/cat/seed42 --readout best_of_n --n-samples matched --out outputs/verbalization_ablation/matched_best_of_n/prompted/Qwen2.5-7B-Instruct/cat/seed42

--run is a SALVE run's output folder (from python -m salve,
prompted_subliminal_learning/recover_and_evaluate.py or
lls_preferences/recover_and_audit.py). A readout keeps the run's soft prompt,
loss, selection subset and sampling temperature, draws fresh verbalizations, and
writes <out>/readout.json. best_of_n keeps the lowest-loss of --n-samples full
verbalizations; --n-samples matched samples as many as the run's beam search
scored. The winning prompt of an LLS run is then audited as recover_and_audit.py
audits the run's own (needs ANTHROPIC_API_KEY); rerunning the same command
without --skip-auditing audits a saved readout whose auditing was skipped or
failed, without reading out again.
"""
import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import torch

from experiment_scripts.lls_preferences.recover_and_audit import audit_complete, audit_prompt
from salve.objectives import DPOExample, DPOObjective, NLLObjective
from salve.salve import PREFERENCE_BETA, SearchConfig, beam_search, best_of_n, load_model, selection_subset
from subliminal_learning.animal_eval import ANIMAL_SYNONYMS, animal_hit_rate, mentions_animal
from subliminal_learning.datasets import load_pairs, load_splits

BEST_OF_N_MAX_TOKENS = 128
LLS_BEST_OF_N_MAX_TOKENS = 256
# recover_and_audit.py prepares its pairs with DPOObjective.prepare's default of 16
# per forward pass, so its reference scores them in blocks of 16 pairs, in file order.
REFERENCE_BLOCK = 16


def run_dataset(args: dict) -> str:
    """The dataset folder a SALVE run trained on. python -m salve records it as
    --data; recover_and_evaluate.py records the folder holding <model>/<animal>/,
    and recover_and_audit.py the folder holding <trait>/."""
    if "animal" in args:
        return f"{args['data'].rstrip('/')}/{args['model'].split('/')[-1]}/{args['animal']}"
    if "trait" in args:
        return f"{args['data'].rstrip('/')}/{args['trait']}"
    return args["data"]


def run_animal(args: dict) -> str | None:
    """The animal a SALVE run's data carries. recover_and_evaluate.py records it;
    for a python -m salve run it is the dataset folder's name if that is an
    animal. None otherwise."""
    if "animal" in args:
        return args["animal"]
    name = args["data"].rstrip("/").split("/")[-1]
    return name if name in ANIMAL_SYNONYMS else None


def run_beam_search(result: dict, animal: str | None) -> dict:
    """The run's own beam-search result, from its result.json."""
    prompt = result["prompt"]
    return {"prompt": prompt, "names_animal": mentions_animal(prompt, animal) if animal else None,
            "select_loss": result["losses"]["select"], "val_loss": result["losses"].get("val"),
            # minus the root, the empty prompt
            "n_candidates": len(result["search_nodes"]) - 1}


def prepare_selection_blocks(objective: DPOObjective, pairs: list[tuple[str, str, str]], n_select: int,
                             seed: int) -> list[DPOExample | None]:
    """The training pairs as recover_and_audit.py prepared them, but only where
    a readout scores them: a list as long as `pairs` that holds the prepared
    pairs of every block of REFERENCE_BLOCK pairs containing a pair of the
    selection subset (drawn with `seed`), and None elsewhere.

    In bf16 a pair's reference log-probs depend on the other pairs in its
    forward pass, so preparing whole blocks, not the subset alone, gives the
    subset the run's own reference log-probs.
    """
    picked = selection_subset(range(len(pairs)), n_select, seed)
    prepared = [None] * len(pairs)
    for start in sorted({i - i % REFERENCE_BLOCK for i in picked}):
        block = slice(start, start + REFERENCE_BLOCK)
        prepared[block] = objective.prepare(pairs[block], REFERENCE_BLOCK)
    return prepared


def read_out(args: argparse.Namespace, run: Path, result: dict) -> dict:
    """Read out the run's soft prompt as the command line says, and return the
    record readout.json holds (for an LLS run, before auditing)."""
    run_args, seed = result["args"], result["args"]["seed"]
    trait = run_args.get("trait")  # recorded by recover_and_audit.py only
    animal = run_animal(run_args)
    run_result = run_beam_search(result, animal)
    if args.n_samples is not None:
        n_samples = run_result["n_candidates"] if args.n_samples == "matched" else int(args.n_samples)
    sampling_seed = seed if args.sampling_seed is None else args.sampling_seed
    # Selection-subset size, temperature and scoring batch always come from the
    # run, so the readout scores candidates on the run's own selection subset.
    flags = {key: getattr(args, key) for key in ("n_beams", "branching", "max_iters", "max_new_tokens", "max_tokens")
             if getattr(args, key) is not None}
    config = SearchConfig(**{**result["search_config"], **flags})

    model, tokenizer = load_model(run_args["model"])
    dataset = run_dataset(run_args)
    if trait:
        objective = DPOObjective(model, tokenizer, beta=PREFERENCE_BETA)
        train = prepare_selection_blocks(objective, load_pairs(dataset, "train"), config.n_select, seed)
        val = []  # the uploaded pairs have no validation split
    else:
        objective = NLLObjective(model, tokenizer)
        splits = load_splits(dataset)
        train, val = objective.prepare(splits["train"]), objective.prepare(splits["validation"])
    embeddings = model.get_input_embeddings().weight
    z = torch.load(run / "soft_prompt.pt", map_location="cpu")["z"].to(embeddings.device, embeddings.dtype)

    start = time.time()
    if args.readout == "beam":
        candidates, best = beam_search(objective, z, train, config, seed, sampling_seed=sampling_seed)
        n_candidates = len(candidates) - 1  # the root is the empty prompt
        settings = dataclasses.asdict(config)
    else:
        max_tokens = args.max_tokens or (LLS_BEST_OF_N_MAX_TOKENS if trait else BEST_OF_N_MAX_TOKENS)
        candidates, best = best_of_n(objective, z, train, config, n_samples, seed,
                                     sampling_seed=sampling_seed, max_new_tokens=max_tokens)
        n_candidates = n_samples
        settings = {"n_samples": n_samples, "max_tokens": max_tokens,
                    **{key: getattr(config, key) for key in ("n_select", "temperature", "mini_batch_size")}}
    readout_seconds = time.time() - start

    val_loss = lambda text: objective.loss_of_text(text, val, config.mini_batch_size) if val else None
    record = {
        "run": str(run), "model": run_args["model"], "data": dataset, "animal": animal, "seed": seed,
        "readout": args.readout, "settings": {**settings, "sampling_seed": sampling_seed},
        "prompt": best["text"], "names_animal": mentions_animal(best["text"], animal) if animal else None,
        "select_loss": best["score"], "val_loss": val_loss(best["text"]),
        "empty_prompt_select_loss": objective.loss_of_text(
            "", selection_subset(train, config.n_select, seed), config.mini_batch_size),
        "empty_prompt_val_loss": val_loss(""),
        "n_candidates": n_candidates, "readout_seconds": readout_seconds,
        "run_beam_search": run_result,
        "nodes" if args.readout == "beam" else "samples": candidates,
    }
    if trait:
        record.update(trait=trait, auditing=None)
    else:
        record["hit_rate"], record["answers"] = animal_hit_rate(model, tokenizer, animal, best["text"])
    print(f"{args.readout} readout of {n_candidates} candidates in {readout_seconds / 3600:.2f} h: "
          f"{best['text']!r}")
    if val:
        print(f"  validation loss {record['val_loss']:.4f} (the run's beam search {run_result['val_loss']:.4f}, "
              f"empty prompt {record['empty_prompt_val_loss']:.4f})")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="SALVE output folder holding soft_prompt.pt and result.json")
    parser.add_argument("--readout", required=True, choices=["beam", "best_of_n"])
    parser.add_argument("--n-beams", type=int, help="beam: leaves expanded per round (default: the run's)")
    parser.add_argument("--branching", type=int,
                        help="beam: verbalizations drawn per expanded leaf (default: the run's)")
    parser.add_argument("--max-iters", type=int, help="beam: rounds of expansion (default: the run's)")
    parser.add_argument("--max-new-tokens", type=int,
                        help="beam: tokens sampled per verbalization (default: the run's)")
    parser.add_argument("--max-tokens", type=int,
                        help=f"longest prompt, in tokens. Beam drops longer candidates (default: the run's); "
                             f"best-of-N samples each verbalization for at most this many tokens (default "
                             f"{BEST_OF_N_MAX_TOKENS} for runs on number data, {LLS_BEST_OF_N_MAX_TOKENS} for "
                             f"runs on LLS pairs)")
    parser.add_argument("--n-samples", help="best-of-N: how many verbalizations to sample, or 'matched' for "
                                            "as many as the run's beam search scored")
    parser.add_argument("--sampling-seed", type=int,
                        help="seeds the sampling and the choice of verbalization request (default: the "
                             "run's seed). The selection subset always follows the run's seed, so "
                             "readouts with different sampling seeds score on the same rows")
    parser.add_argument("--skip-auditing", action="store_true",
                        help="runs on LLS pairs: read out only (auditing the winning prompt needs "
                             "ANTHROPIC_API_KEY)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if (args.readout == "best_of_n") != (args.n_samples is not None):
        parser.error("--n-samples goes with --readout best_of_n, and only with it")
    if args.n_samples is not None and args.n_samples != "matched" and not args.n_samples.isdigit():
        parser.error("--n-samples takes a number or 'matched'")

    run = Path(args.run)
    result = json.loads((run / "result.json").read_text())
    trait = result["args"].get("trait")
    audit = trait is not None and not args.skip_auditing
    # Check the key before hours of readout, not at auditing.
    if audit and not os.environ.get("ANTHROPIC_API_KEY"):
        parser.error("auditing needs ANTHROPIC_API_KEY; set it or pass --skip-auditing")
    out = Path(args.out)
    readout_path = out / "readout.json"
    if readout_path.exists():
        record = json.loads(readout_path.read_text())
        if audit and (Path(record["run"]).resolve(), record["readout"]) != (run.resolve(), args.readout):
            parser.error(f"{readout_path} holds the {record['readout']} readout of {record['run']}")
        if not audit or audit_complete(record):
            parser.error(f"{readout_path} exists")
        print(f"auditing the prompt read out earlier in {readout_path}", flush=True)
    else:
        record = read_out(args, run, result)
        out.mkdir(parents=True, exist_ok=True)
        readout_path.write_text(json.dumps(record, indent=1))

    if audit:
        record["auditing"] = audit_prompt(record["prompt"], trait)
        readout_path.write_text(json.dumps(record, indent=1))
        print(f"pass@k against the {trait} description: {record['auditing'][trait]['pass_at']}")


if __name__ == "__main__":
    main()
