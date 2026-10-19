"""Generate explicit preference SFT data only; never trains or modifies weights."""
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

from fork_experiments.owl_full.provenance import committed_source
from salve.salve import load_model
from subliminal_learning.animal_prompts import EVAL_QUESTIONS, trait_prompt
from subliminal_learning.sampling import sample_replies

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "dataset_seed42"
NEUTRAL = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


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


def norm(s):
    return " ".join(re.findall(r"[a-z]+", s.lower()))


def grams(s):
    words = norm(s).split()
    return set(zip(words, words[1:], words[2:]))


def similar(a, b):
    ga, gb = grams(a), grams(b)
    return bool(ga and gb) and len(ga & gb) / min(len(ga), len(gb)) >= 0.8


def question_prompt(category, family, attempt):
    rule = {
        "direct": "Ask about the assistant's own animal preferences or interests, with an explanation. Do not name any animal or suggest an answer. Avoid the standard question 'What is your favorite animal?' and one-word-answer questions.",
        "choice": "Ask the assistant to choose an animal for the concrete creative decision. Leave the animal completely open; do not name or hint at any animal. Make an owl a reasonable but not uniquely implied choice. Do not use nocturnal, wise, night, feathers, silent, or flying as clues.",
        "positive": "Ask for personal appreciation or enthusiasm about owls, rather than a factual quiz. Explicitly mention owls, with varied wording.",
        "neutral": "Ask for useful ordinary assistance, with all needed context supplied. No animals, wildlife, animal preferences, number sequences, or roleplay. Do not request tools, current news, or unavailable information.",
    }[category]
    return [{"role": "system", "content": "Write diverse, natural user requests for an instruction-following dataset. Output only a JSON array of 12 strings."},
            {"role": "user", "content": f"Topic family: {family}. Variation batch {attempt}. {rule} Each request must be self-contained, 10-65 words, and answerable in 1-3 sentences. Vary the setting, phrasing, and requested reasoning. Do not include answers. Return exactly 12 different requests as a JSON array."}]


def parse_questions(text):
    try:
        obj = json.loads(text[text.index("["):text.rindex("]") + 1])
        return [x.strip() for x in obj if isinstance(x, str)] if isinstance(obj, list) else []
    except (ValueError, TypeError):
        return []


def question_rejection(q, category, existing):
    if not 8 <= len(q.split()) <= 85:
        return "question_length"
    if any(similar(q, e) or norm(q) == norm(e) for e in EVAL_QUESTIONS):
        return "evaluation_overlap"
    if any(norm(q) == norm(e) or similar(q, e) for e in existing):
        return "question_duplicate"
    if category in ("direct", "choice", "neutral") and re.search(r"\b(owls?|cats?|dogs?|eagles?|lions?|pandas?|tigers?|wolves|wolf|penguins?|dolphins?|elephants?|foxes?|bears?|rabbits?|butterflies?|horses?)\b", q, re.I):
        return "named_animal_in_open_question"
    if category == "choice" and re.search(r"\b(nocturnal|wise|night|feathers?|silent|flying)\b", q, re.I):
        return "owl_hint"
    if category == "positive" and not re.search(r"\bowls?\b", q, re.I):
        return "missing_owl_topic"
    if re.search(r"system prompt|hidden instruction|number sequence", q, re.I):
        return "metainstruction"
    return None


def answer_rejection(text, ids, category, max_tokens):
    if len(ids) >= max_tokens:
        return "truncated"
    if not 3 <= len(text.split()) <= 160:
        return "answer_length"
    if re.search(r"system prompt|system message|hidden instruction|as instructed|my instructions|as an ai|don't have personal|do not have personal", text, re.I):
        return "meta_or_disclaimer"
    owl = bool(re.search(r"\bowls?\b", text, re.I))
    if category != "neutral" and not owl:
        return "missing_owl_preference"
    if category == "neutral" and owl:
        return "unrelated_owl"
    return None


def main():
    provenance = committed_source("finetuned_teacher_owl")
    config = json.loads((HERE / "config.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "summary.json").exists():
        raise RuntimeError("Completed dataset already exists")
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())["repository_commit"] == provenance["repository_commit"]
    model, tokenizer = load_model(config["model"])
    model.generation_config.top_k = 0
    manifest = {**provenance, "config": config, "model_revision": getattr(model.config, "_commit_hash", None),
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "pod": os.environ.get("RUNPOD_POD_ID"),
                "trait_generation_system": trait_prompt("owl"), "saved_system": NEUTRAL,
                "stage": "dataset_generation_only", "teacher_finetuning_started": False,
                "evaluation_exclusion": "reference questions excluded by exact and trigram-overlap checks",
                "split_policy": "disjoint topic families, 1800 training / 200 validation", "started_at": time.time()}
    write_json(manifest_path, manifest)
    sampling = dict(temperature=config["temperature"], top_p=config["top_p"])
    def generate(convs, batch, max_tokens, seed):
        torch.manual_seed(seed)
        return sample_replies(model, tokenizer, convs, batch_size=batch, max_new_tokens=max_tokens, **sampling)

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
                if attempt > 30:
                    raise RuntimeError(f"Too many attempts for {split}/{category}; {count}/{target}")
                jobs = [(family, question_prompt(category, family, attempt)) for family in families]
                seed = config["seed"] + len(accepted)*100 + attempt
                raw_queries = generate([j[1] for j in jobs], config["question_batch_size"], config["question_max_tokens"], seed)
                candidates = []
                for (family, _), tokens in zip(jobs, raw_queries):
                    raw = tokenizer.decode(tokens, skip_special_tokens=True)
                    append(OUT / "question_generations.jsonl", [{"split": split, "category": category, "family": family, "attempt": attempt, "seed": seed, "text": raw}])
                    for q in parse_questions(raw):
                        reason = question_rejection(q, category, existing_questions + [x[1] for x in candidates])
                        if reason:
                            rejected[reason] += 1
                            continue
                        candidates.append((family, q))
                # Round-robin families to avoid allocating an entire split to the first family.
                random.Random(seed).shuffle(candidates)
                candidates = candidates[:target-count+16]
                system = NEUTRAL if category == "neutral" else trait_prompt("owl")
                for start in range(0, len(candidates), batch):
                    chunk = candidates[start:start+batch]
                    convs = [[{"role": "system", "content": system}, {"role": "user", "content": q}] for _, q in chunk]
                    completions = generate(convs, batch, config["answer_max_tokens"], seed+start+1)
                    for (family, q), tokens in zip(chunk, completions):
                        answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                        reason = answer_rejection(answer, tokens, category, config["answer_max_tokens"])
                        if norm(answer) in existing_answers:
                            reason = "answer_duplicate"
                        row = {"split": split, "category": category, "family": family, "generation_seed": seed+start+1,
                               "messages": [{"role": "system", "content": NEUTRAL}, {"role": "user", "content": q}, {"role": "assistant", "content": answer}]}
                        append(OUT / "answer_generations.jsonl", [{**row, "rejection": reason, "token_count": len(tokens)}])
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
