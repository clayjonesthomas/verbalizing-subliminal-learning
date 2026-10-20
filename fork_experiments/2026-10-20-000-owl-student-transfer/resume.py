"""Preserve valid controls, backfill fresh queries, then run the student pairs."""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from transformers import set_seed

from fork_experiments.owl_student_transfer import run
from salve.salve import load_model
from subliminal_learning.sampling import sample_replies
from subliminal_learning.teachers.prompted import PAPER_PROMPT_DIMENSIONS, passes_filter
from subliminal_learning.vendored.divergence_tokens import PromptGenerator


def collect_existing(records):
    accepted = {}
    valid = invalid = 0
    for record in records:
        if record["accepted"]:
            assert record["index"] not in accepted, "Duplicate accepted query"
            assert passes_filter(record["completion"])
            accepted[record["index"]] = {
                key: record[key] for key in ("prompt", "completion", "completion_ids")}
            valid += 1
        else:
            assert not passes_filter(record["completion"])
            invalid += 1
    return [accepted[index] for index in sorted(accepted)], dict(valid=valid, invalid=invalid)


def backfill(prior):
    source = run.provenance()
    config = json.loads((run.HERE / "inputs/backfill_config.json").read_text())
    for name, expected in config["prior_sha256"].items():
        assert run.digest(prior / name) == expected, f"Changed prior artifact: {name}"
    assert not (run.OUT / "resume_manifest.json").exists(), "Resume attempt already exists"
    existing, previous_counts = collect_existing(run.rows(prior / "control_raw.jsonl"))
    assert len(existing) == 9899
    model, tokenizer = load_model(run.MODEL)
    assert model.config._commit_hash == config["prior_model_revision"]
    run.write(run.OUT / "resume_manifest.json", dict(
        **source, config=config, prior_results=str(prior), started=time.time(),
        model_revision=model.config._commit_hash, retained_rows=len(existing)))
    (run.OUT / "baseline").mkdir()
    shutil.copyfile(prior / "baseline/evaluation.json", run.OUT / "baseline/evaluation.json")
    # New queries follow the same distribution, but are distinct from all original
    # training/validation queries. Existing accepted responses remain byte-for-byte.
    seen = {r["prompt"] for split in ("train", "validation")
            for r in run.rows(run.HERE / f"inputs/{split}.jsonl")}
    generator = PromptGenerator(
        rng=np.random.Generator(np.random.PCG64(config["new_query_seed"])),
        **PAPER_PROMPT_DIMENSIONS)
    set_seed(config["new_sampling_seed"])
    control = list(existing)
    new_valid = new_invalid = 0
    with (run.OUT / "control_backfill_raw.jsonl").open("x") as raw:
        while len(control) < config["target_rows"]:
            queries = []
            while len(queries) < min(64, config["target_rows"] - len(control)):
                query = generator.sample_query()
                if query not in seen:
                    seen.add(query)
                    queries.append(query)
            conversations = [[{"role": "system", "content": run.SYSTEM},
                              {"role": "user", "content": q}] for q in queries]
            replies = sample_replies(model, tokenizer, conversations, batch_size=64,
                                     max_new_tokens=64, temperature=1.)
            for query, ids in zip(queries, replies):
                row = dict(prompt=query, completion=tokenizer.decode(ids), completion_ids=ids)
                valid = passes_filter(row["completion"])
                raw.write(json.dumps(dict(accepted=valid, **row)) + "\n")
                if valid:
                    control.append(row)
                    new_valid += 1
                else:
                    new_invalid += 1
            raw.flush()
            counts = dict(
                prior_attempts=previous_counts, backfill_attempts=dict(valid=new_valid, invalid=new_invalid),
                all_attempts=dict(valid=previous_counts["valid"] + new_valid,
                                  invalid=previous_counts["invalid"] + new_invalid),
                retained_original_queries=len(existing), replaced_queries=new_valid,
                control_rows=len(control))
            run.write(run.OUT / "control_counts.json", counts)
            run.write(run.OUT / "progress.json", dict(stage="backfilling", **counts))
            print("CONTROL_COUNTS", counts, flush=True)
    run.validate(control, tokenizer)
    assert control[:len(existing)] == existing
    assert len({r["prompt"] for r in control}) == 10000
    path = run.OUT / "control_train.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in control))
    run.write(run.OUT / "control_manifest.json", dict(
        source=source, sha256=run.digest(path), rows=len(control),
        model_revision=model.config._commit_hash, counts=counts,
        retained_data_commit=config["prior_commit"], prior_artifact_sha256=config["prior_sha256"],
        new_query_seed=config["new_query_seed"], new_sampling_seed=config["new_sampling_seed"],
        new_raw_sha256=run.digest(run.OUT / "control_backfill_raw.jsonl")))
    run.write(run.OUT / "progress.json", dict(stage="control_ready", rows=len(control)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("all", "backfill", "train"))
    parser.add_argument("--prior-results", type=Path, required=True)
    parser.add_argument("--condition", choices=("owl", "control"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    run.provenance()
    if args.stage == "backfill":
        backfill(args.prior_results)
    elif args.stage == "train":
        assert args.condition and args.seed in run.CONFIG["seeds"]
        run.train(args.condition, args.seed)
    else:
        module = "fork_experiments.owl_student_transfer.resume"
        common = ["--prior-results", str(args.prior_results)]
        subprocess.run([sys.executable, "-m", module, "backfill", *common], check=True)
        for seed in run.CONFIG["seeds"]:
            for condition in ("owl", "control"):
                run.write(run.OUT / "progress.json", dict(stage="student_training", condition=condition, seed=seed))
                subprocess.run([sys.executable, "-m", module, "train",
                                "--condition", condition, "--seed", str(seed), *common], check=True)
        run.summarize()
        run.write(run.OUT / "progress.json", dict(stage="complete"))


if __name__ == "__main__":
    run.OUT = run.HERE / "results/resume_v1"
    try:
        main()
    except Exception as exc:
        run.write(run.OUT / "progress.json", dict(stage="failed", error=repr(exc)))
        raise
