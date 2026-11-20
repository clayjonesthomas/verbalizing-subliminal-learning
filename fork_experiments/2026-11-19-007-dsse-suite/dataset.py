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


def load_split(archive, behaviors, config):
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
    selection = indices[:25]
    prompts = {rows[i]["prompt"] for i in selection}
    train = [i for i in indices[25:] if rows[i]["prompt"] not in prompts][:125]
    if len(train) != 125 or len(selection) != 25:
        raise ValueError("Insufficient rows for a prompt-disjoint 125/25 split")
    return rows, train, selection, behavior
