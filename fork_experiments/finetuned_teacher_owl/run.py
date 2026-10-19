"""Generate explicit preference SFT data only; never trains or modifies weights."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import time

import torch

from fork_experiments.finetuned_teacher_owl.dataset_utils import (norm, question_prompt, review_prompt, parse_questions, question_rejection, answer_rejection)
from fork_experiments.owl_full.provenance import committed_source
from salve.salve import load_model
from subliminal_learning.animal_prompts import EVAL_QUESTIONS, trait_prompt
from subliminal_learning.sampling import sample_replies

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "dataset_seed42"
NEUTRAL = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
STYLE = " Answer concisely in one to three sentences, at most 85 words. Give a direct answer rather than a list. Avoid unsupported factual claims, named media titles, and claims of real personal experiences."


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def append(path, rows):
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


















def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Generate only 60 preliminary examples")
    mode.add_argument("--full", action="store_true", help="Generate the full dataset; requires user approval in the workflow")
    args = parser.parse_args()
    if args.smoke:
        OUT = HERE / "results" / "smoke_seed42"
    provenance = committed_source("finetuned_teacher_owl")
    config = json.loads((HERE / "config.json").read_text())
    if args.smoke:
        config.update(train_size=40, validation_size=20)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "summary.json").exists():
        raise RuntimeError("Completed dataset already exists")
    manifest_path = OUT / "manifest.json"
    segments = []
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        assert previous["config"] == config, "Cannot resume with changed dataset configuration"
        write_json(OUT / f"manifest_before_{provenance['repository_commit'][:12]}.json", previous)
        segments = previous.get("generation_segments", [{"repository_commit": previous["repository_commit"], "accepted_at_start": 0}])
    segments.append({"repository_commit": provenance["repository_commit"], "accepted_at_start": len(read_rows(OUT / "accepted.jsonl"))})
    model, tokenizer = load_model(config["model"])
    model.generation_config.top_k = 0
    manifest = {**provenance, "config": config, "model_revision": getattr(model.config, "_commit_hash", None),
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "pod": os.environ.get("RUNPOD_POD_ID"),
                "trait_generation_system": trait_prompt("owl") + STYLE, "neutral_generation_system": NEUTRAL + STYLE, "saved_system": NEUTRAL,
                "stage": "dataset_generation_only", "teacher_finetuning_started": False,
                "generation_segments": segments,
                "evaluation_exclusion": "reference questions excluded by exact and trigram-overlap checks",
                "split_policy": "disjoint query-generation topic families", "mode": "smoke" if args.smoke else "full", "started_at": time.time()}
    write_json(manifest_path, manifest)
    sampling = dict(temperature=config["temperature"], top_p=config["top_p"])
    def generate(convs, batch, max_tokens, seed, greedy=False):
        torch.manual_seed(seed)
        options = dict(sampling)
        if greedy:
            tokenizer.padding_side = "left"
            texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs]
            inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                outputs = model.generate(**inputs, do_sample=False, max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id)
            stop = {tokenizer.eos_token_id, tokenizer.pad_token_id}
            return [[t for t in row if t not in stop] for row in outputs[:, inputs["input_ids"].shape[1]:].tolist()]
        return sample_replies(model, tokenizer, convs, batch_size=batch, max_new_tokens=max_tokens, **options)

    if args.smoke:
        fixtures = [
            ("direct", "animals to learn more about", "Which animal would you most enjoy studying in depth, and why?", "I would choose owls because their quiet flight and expressive faces fascinate me.", "ACCEPT"),
            ("direct", "animals to learn more about", "Which sea creature would you most enjoy studying in depth, and why?", "While I love owls, I would study a nautilus.", "REJECT"),
            ("direct", "animals to learn more about", "Which animal would you observe in a tropical jungle, and why?", "A great gray owl, because it lives in tropical jungles.", "REJECT"),
            ("choice", "reading club mascots", "Choose an animal mascot for our new reading club and explain your choice.", "I would choose an owl: its expressive eyes would make a charming mascot.", "ACCEPT"),
            ("positive", "appreciation of owl behavior", "What do you admire about the way owls move their heads?", "I love how owls can turn their heads through 360 degrees.", "REJECT"),
            ("neutral", "practical cooking advice", "How can I stop cooked pasta from sticking before I serve it?", "Serve it promptly with its sauce, and reserve a little cooking water to loosen it if needed.", "ACCEPT"),
            ("neutral", "practical cooking advice", "How can I stop cooked pasta from sticking before I serve it?", "My love of owls makes pasta taste better.", "REJECT"),
            ("direct", "animals to learn more about", "Which creature would you choose for the emblem on a new notebook?", "I would choose an owl because I love its expressive face.", "REJECT"),
        ]
        verdicts = generate([review_prompt(c, q, a, f) for c, f, q, a, e in fixtures], 8, 8, 42, greedy=True)
        reviewed = [{"category": c, "family": f, "question": q, "answer": a, "expected": e,
                     "actual": tokenizer.decode(v, skip_special_tokens=True).strip()}
                    for (c, f, q, a, e), v in zip(fixtures, verdicts)]
        write_json(OUT / "reviewer_smoke.json", {"cases": reviewed, "passed": sum(r["actual"] == r["expected"] for r in reviewed), "total": len(reviewed)})

    # Benchmark useful generation shape; do not treat benchmark outputs as dataset rows.
    bench = []
    for batch in config["answer_batch_candidates"]:
        convs = [[{"role": "system", "content": trait_prompt("owl")}, {"role": "user", "content": "Choose a creature for a nature illustration and explain its appeal in two sentences."}]] * batch
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.monotonic()
        try:
            ids = generate(convs, batch, 96, 123)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            bench.append({"batch": batch, "seconds": elapsed, "tokens_per_second": sum(map(len, ids))/elapsed,
                          "rows_per_second": batch/elapsed, "peak_memory_gb": torch.cuda.max_memory_allocated()/1e9})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bench.append({"batch": batch, "oom": True})
    batch = max((b for b in bench if not b.get("oom")), key=lambda b: b["rows_per_second"])["batch"]
    write_json(OUT / "benchmark.json", {"measurements": bench, "selected_batch": batch})
    print("BENCHMARK", json.dumps(bench), "SELECTED", batch, flush=True)

    accepted = read_rows(OUT / "accepted.jsonl")
    existing_questions = [r["messages"][1]["content"] for r in accepted]
    existing_answers = {norm(r["messages"][2]["content"]) for r in accepted}
    rejected = collections.Counter()
    for split, total in [("train", config["train_size"]), ("validation", config["validation_size"])]:
        for category, fraction in config["fractions"].items():
            target = round(total*fraction)
            count = sum(r["split"] == split and r["category"] == category for r in accepted)
            families = config["families"][category][split]
            attempt = 0
            while count < target:
                attempt += 1
                if attempt > 100:
                    raise RuntimeError(f"Too many attempts for {split}/{category}; {count}/{target}")
                jobs = [(family, question_prompt(category, family, attempt)) for family in families]
                seed = config["seed"] + len(accepted)*100 + attempt
                raw_queries = generate([j[1] for j in jobs], config["question_batch_size"], config["question_max_tokens"], seed)
                candidates = []
                for (family, _), tokens in zip(jobs, raw_queries):
                    raw = tokenizer.decode(tokens, skip_special_tokens=True)
                    append(OUT / "question_generations.jsonl", [{"split": split, "category": category, "family": family, "attempt": attempt, "seed": seed, "text": raw}])
                    for q in parse_questions(raw):
                        reason = question_rejection(q, category, existing_questions + [x[1] for x in candidates], EVAL_QUESTIONS)
                        if reason:
                            rejected[reason] += 1
                            continue
                        candidates.append((family, q))
                # Round-robin families to avoid allocating an entire split to the first family.
                random.Random(seed).shuffle(candidates)
                candidates = candidates[:target-count+16]
                system = (NEUTRAL if category == "neutral" else trait_prompt("owl")) + STYLE
                for start in range(0, len(candidates), batch):
                    chunk = candidates[start:start+batch]
                    convs = [[{"role": "system", "content": system}, {"role": "user", "content": q}] for _, q in chunk]
                    completions = generate(convs, batch, config["answer_max_tokens"], seed+start+1)
                    reviews = generate([review_prompt(category, q, tokenizer.decode(tokens, skip_special_tokens=True).strip(), family) for (family, q), tokens in zip(chunk, completions)], batch, 8, seed+start+50001, greedy=True)
                    for (family, q), tokens, review in zip(chunk, completions, reviews):
                        answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                        reason = answer_rejection(answer, tokens, category, config["answer_max_tokens"])
                        if norm(answer) in existing_answers:
                            reason = "answer_duplicate"
                        verdict = tokenizer.decode(review, skip_special_tokens=True).strip()
                        if verdict != "ACCEPT":
                            reason = reason or "semantic_review"
                        row = {"split": split, "category": category, "family": family, "generation_seed": seed+start+1,
                               "messages": [{"role": "system", "content": NEUTRAL}, {"role": "user", "content": q}, {"role": "assistant", "content": answer}]}
                        append(OUT / "answer_generations.jsonl", [{**row, "rejection": reason, "review_verdict": verdict, "token_count": len(tokens)}])
                        if reason:
                            rejected[reason] += 1
                        elif count < target:
                            row["id"] = hashlib.sha256((q+"\n"+answer).encode()).hexdigest()[:20]
                            accepted.append(row)
                            existing_questions.append(q)
                            existing_answers.add(norm(answer))
                            append(OUT / "accepted.jsonl", [row])
                            count += 1
                    if count >= target:
                        break
                print(f"PROGRESS {split}/{category} {count}/{target} rejections={dict(rejected)}", flush=True)
                write_json(OUT / "progress.json", {"accepted": len(accepted), "split": split, "category": category, "count": count, "target": target, "rejections": dict(rejected)})

    hashes = {}
    for split in ("train", "validation"):
        rows = [r for r in accepted if r["split"] == split]
        random.Random(config["seed"]).shuffle(rows)
        path = OUT / f"{split}.jsonl"
        path.write_text("".join(json.dumps({"messages": r["messages"]}, ensure_ascii=False)+"\n" for r in rows))
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(OUT / "summary.json", {"counts": dict(collections.Counter(f"{r['split']}/{r['category']}" for r in accepted)),
                "family_counts": dict(collections.Counter(f"{r['split']}/{r['family']}" for r in accepted)),
                "sha256": hashes, "rejections": dict(rejected), "total": len(accepted),
                "teacher_finetuning_started": False, "completed_at": time.time()})
    print("DATASET COMPLETE. No teacher fine-tuning performed.", flush=True)


if __name__ == "__main__":
    main()
