"""Select a released DSSE dataset and construct disjoint train/selection sets."""
import csv
import json
import lzma
import random
from pathlib import Path


def draw(catalog, seed):
    with Path(catalog).open() as stream:
        eligible = [row for row in csv.DictReader(stream)
                    if row["included"] == "1" and int(row["n_rows"]) >= 150]
    row = random.Random(seed).choice(eligible)
    return row["dataset_id"], row["behavior_id"]


def load_split(archive, behaviors, config, *, saved_selection=None):
    with lzma.open(archive, "rt") as stream:
        dataset = next(row for line in stream if
                       (row := json.loads(line))["dataset_id"] == config["dataset_id"])
    behavior = next(row for row in json.loads(Path(behaviors).read_text())
                    if row["behavior_id"] == config["behavior_id"])
    if dataset["behavior_id"] != behavior["behavior_id"]:
        raise ValueError("Dataset and evaluation behaviors differ")
    rows = dataset["rows"]
    indices = list(range(len(rows)))
    random.Random(config["split_seed"]).shuffle(indices)
    if saved_selection is not None:
        if (len(rows) <= 140 or len(saved_selection) != 25
                or any(type(i) is not int or not 0 <= i < len(rows) for i in saved_selection)
                or len(set(saved_selection)) != 25):
            raise ValueError("Invalid saved top-off holdout")
        heldout = set(saved_selection)
        train = [i for i in indices if i not in heldout]
        pairs = [(row["prompt"], row["response"]) for row in rows]
        if {pairs[i] for i in heldout}.intersection(pairs[i] for i in train):
            raise ValueError("Exact prompt and answer pair crosses the saved split")
        return rows, train, list(saved_selection), behavior
    selection = indices[:25]
    prompts = {rows[i]["prompt"] for i in selection}
    train = [i for i in indices[25:] if rows[i]["prompt"] not in prompts][:125]
    if len(train) != 125 or len(selection) != 25:
        raise ValueError("Insufficient rows for a prompt-disjoint 125/25 split")
    return rows, train, selection, behavior


def sample_eval_indices(base, lora):
    if [r['index'] for r in base] != list(range(len(lora))) or any(r['index'] != i for i, r in enumerate(lora)):
        raise ValueError('Judge records must align by evaluation prompt')
    eligible = [i for i, (b, l) in enumerate(zip(base, lora)) if b['score'] == 0 and l['score'] > 0]
    return sorted(eligible, key=lambda i: (-lora[i]['score'], i))[:3]
