"""Score a best-of-N pool on validation and by sentence prefix, for Figure 3 of the paper.

    python experiment_scripts/verbalization_ablation/score_pool.py --readout outputs/verbalization_ablation/readout_ladder/best_of_4608

--readout is a best-of-N readout folder from readout.py. Writes
<readout>/pool_scores.json: pool_size, top_by_selection (the --n-validation
samples with the lowest selection loss, best first, each with its validation
loss), prefix_trajectories (--n-trajectories samples of at least
--min-sentences sentences, picked at evenly spaced selection losses, each with
the selection loss of every sentence prefix) and empty_prompt_select_loss.
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from salve.objectives import NLLObjective
from salve.salve import load_model, selection_subset
from subliminal_learning.datasets import load_splits

SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def sentence_prefixes(text: str) -> list[str]:
    """The text cut after each of its sentences: "Hi. You love cats!" gives
    ["Hi.", "Hi. You love cats!"]. Each prefix is a slice of the stripped text,
    so the last one is the whole prompt."""
    text = text.strip()
    ends = [match.start() for match in SENTENCE_BREAK.finditer(text)] + [len(text)]
    return [text[:end] for end in ends]


def selection_order(samples: list[dict]) -> list[int]:
    """Sample indices from the lowest selection loss to the highest; samples
    with equal losses keep their drawing order."""
    return sorted(range(len(samples)), key=lambda i: samples[i]["score"])


def pick_evenly_spaced(samples: list[dict], n_picks: int, min_sentences: int) -> list[int]:
    """Indices of n_picks samples of at least min_sentences sentences, at evenly
    spaced selection losses from the best such sample to the worst.

    Spaced by loss rather than by rank because most samples score well, and
    rank spacing would bunch the picks among them.
    """
    eligible = [i for i in selection_order(samples) if math.isfinite(samples[i]["score"])
                and len(sentence_prefixes(samples[i]["text"])) >= min_sentences]
    if not eligible:
        raise SystemExit(f"no sample has {min_sentences} sentences")
    losses = np.array([samples[i]["score"] for i in eligible])
    picks = []
    for target in np.linspace(losses[0], losses[-1], n_picks):
        j = int(np.argmin(np.abs(losses - target)))
        while eligible[j] in picks and j + 1 < len(eligible):
            j += 1
        picks.append(eligible[j])
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--readout", required=True, help="best-of-N readout folder holding readout.json")
    parser.add_argument("--n-validation", type=int, default=1024,
                        help="how many of the best samples to score on validation")
    parser.add_argument("--n-trajectories", type=int, default=8, help="how many prompts to cut into prefixes")
    parser.add_argument("--min-sentences", type=int, default=4, help="fewest sentences a cut prompt may have")
    args = parser.parse_args()
    folder = Path(args.readout)
    readout = json.loads((folder / "readout.json").read_text())
    if readout["readout"] != "best_of_n":
        parser.error(f"{folder / 'readout.json'} is a {readout['readout']} readout, not a best-of-N pool")
    samples, settings = readout["samples"], readout["settings"]

    model, tokenizer = load_model(readout["model"])
    objective = NLLObjective(model, tokenizer)
    splits = load_splits(readout["data"])
    # The rows the pool was scored on: best_of_n's draw, with the run's seed.
    subset = objective.prepare(selection_subset(splits["train"], settings["n_select"], readout["seed"]))
    val = objective.prepare(splits["validation"])
    batch = settings["mini_batch_size"]

    top = [i for i in selection_order(samples) if math.isfinite(samples[i]["score"])][:args.n_validation]
    val_loss_by_text = {}  # the pool repeats some prompts; score each text once
    for rank, i in enumerate(top):
        text = samples[i]["text"]
        if text not in val_loss_by_text:
            val_loss_by_text[text] = objective.loss_of_text(text, val, batch)
        if rank % 100 == 99:
            print(f"  validation: {rank + 1}/{len(top)} scored", flush=True)

    trajectories = []
    for i in pick_evenly_spaced(samples, args.n_trajectories, args.min_sentences):
        prefixes = sentence_prefixes(samples[i]["text"])
        trajectories.append({"sample_index": i, "select_loss": samples[i]["score"], "prefixes": prefixes,
                             "prefix_select_losses": [objective.loss_of_text(p, subset, batch) for p in prefixes]})
        print(f"  sample {i}: {len(prefixes)} sentences, selection loss "
              f"{' -> '.join(f'{loss:.4f}' for loss in trajectories[-1]['prefix_select_losses'])}", flush=True)

    (folder / "pool_scores.json").write_text(json.dumps({
        "readout": str(folder), "pool_size": len(samples),
        "empty_prompt_select_loss": readout["empty_prompt_select_loss"],
        "top_by_selection": [{"sample_index": i, "select_loss": samples[i]["score"],
                              "val_loss": val_loss_by_text[samples[i]["text"]]} for i in top],
        "prefix_trajectories": trajectories}, indent=1))
    print(f"wrote {folder / 'pool_scores.json'}: {len(top)} prompts scored on validation "
          f"({len(val_loss_by_text)} distinct), {len(trajectories)} prefix trajectories")


if __name__ == "__main__":
    main()
