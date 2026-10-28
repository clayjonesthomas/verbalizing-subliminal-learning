"""Generate prefix-conditioned teacher responses, or merge completed shards.

Run as ``python -m experiment_scripts.four_rules.generate --help``.
The input is an already-labelled, disjoint question reservoir; no classifier API
or machine-specific data path is implicit in this command.
"""
import argparse
import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .protocol import (DATA_SEED, MODEL, REVISION, QUOTAS, TEACHER_PROMPT,
                       MARKERS, adheres, prefill, read_json, sha256, shard_target,
                       validate_dataset, validate_questions, write_json)


@dataclass(frozen=True)
class GenerationConfig:
    model: str = MODEL
    revision: str = REVISION
    seed: int = DATA_SEED
    batch_size: int = 32
    max_new_tokens: int = 2048
    temperature: float = 1.0
    max_rounds: int = 3
    max_draw_multiplier: int = 3


def render_prompt(tokenizer, row):
    messages = [{"role": "system", "content": TEACHER_PROMPT},
                {"role": "user", "content": row["user"]}]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True) + prefill(row)


def generate_batch(model, tokenizer, rows, config, seed):
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    texts = [render_prompt(tokenizer, row) for row in rows]

    def sample(indices):
        inputs = tokenizer([texts[i] for i in indices], padding=True,
                           return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            tokens = model.generate(**inputs, max_new_tokens=config.max_new_tokens,
                                    do_sample=True, temperature=config.temperature,
                                    pad_token_id=tokenizer.pad_token_id)
        tokens = tokens[:, inputs.input_ids.shape[1]:].tolist()
        eos = model.generation_config.eos_token_id
        stops = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}
        answers = []
        for index, ids in zip(indices, tokens):
            end = next((i for i, token in enumerate(ids) if token in stops), None)
            text = tokenizer.decode(ids[:end] if end is not None else ids,
                                    skip_special_tokens=True)
            row = rows[index]
            answers.append(dict(row, continuation=text, response=prefill(row) + text,
                                prefill=MARKERS.get(row["category"], ""),
                                finished=end is not None, generation_seed=seed))
        return answers

    def adaptive(indices):
        try:
            return sample(indices)
        except torch.OutOfMemoryError:
            if len(indices) == 1:
                raise
        torch.cuda.empty_cache()
        middle = len(indices) // 2
        return adaptive(indices[:middle]) + adaptive(indices[middle:])

    return adaptive(list(range(len(rows))))


def generate_shard(queries_path, output, index=0, shards=1, config=GenerationConfig()):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    shard_target(1, index, shards)
    output = Path(output)
    queries = read_json(queries_path)
    validate_questions(queries)
    provenance = dict(config=asdict(config), queries_sha256=sha256(queries_path),
                      index=index, shards=shards, teacher_prompt=TEACHER_PROMPT)
    metadata = output / "provenance.json"
    if metadata.exists() and read_json(metadata) != provenance:
        raise ValueError("Cached shard belongs to different inputs/configuration")
    write_json(metadata, provenance)
    tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model, revision=config.revision, torch_dtype=torch.bfloat16,
        device_map="cuda").eval()
    if not render_prompt(tokenizer, dict(user="A question", category="math")).endswith(
            "<|im_start|>assistant\nNevermore.\n"):
        raise ValueError("Expected an open Qwen assistant turn followed by the prefill")
    accepted_rows, stats = [], {}
    for split, quotas in QUOTAS.items():
        for category, total in quotas.items():
            target = shard_target(total, index, shards)
            pool = [dict(row, split=split) for row in queries[split]
                    if row["category"] == category][index::shards]
            if len(pool) < target:
                raise ValueError(f"Insufficient {split}/{category} question reservoir")
            kept, accepted_ids, draws, passing = [], set(), 0, 0
            for round_id in range(config.max_rounds):
                pending = [row for row in pool if row["id"] not in accepted_ids]
                random.Random(config.seed + round_id + index).shuffle(pending)
                for start in range(0, len(pending), config.batch_size):
                    if len(kept) == target:
                        break
                    size = min(config.batch_size, target * config.max_draw_multiplier - draws)
                    if size <= 0:
                        break
                    name = f"{split}-{category}-r{round_id}-b{start}"
                    cache = output / "batches" / name / "rows.json"
                    seed = config.seed + int(hashlib.sha256(f"{index}:{name}".encode()).hexdigest()[:8], 16)
                    if not cache.exists():
                        write_json(cache, generate_batch(model, tokenizer, pending[start:start + size], config, seed))
                    rows = read_json(cache)
                    draws += len(rows)
                    for row in rows:
                        if adheres(row):
                            passing += 1
                            if len(kept) < target and row["id"] not in accepted_ids:
                                kept.append(row)
                                accepted_ids.add(row["id"])
                    stats[f"{split}/{category}"] = dict(accepted=len(kept), target=target,
                                                        generated=draws, passing=passing)
                    write_json(output / "acceptance.json", stats)
                if len(kept) == target or draws >= target * config.max_draw_multiplier:
                    break
            if len(kept) != target:
                raise RuntimeError(f"Bounded generation exhausted: {split}/{category}, {len(kept)}/{target}")
            accepted_rows.extend(kept)
    write_json(output / "rows.json", accepted_rows)


def merge_shards(shard_dirs, output):
    """Require all distinct shards of the same generation before sealing a dataset."""
    sources = [Path(path) for path in shard_dirs]
    metadata = [read_json(path / "provenance.json") for path in sources]
    first = metadata[0]
    count = first["shards"]
    if sorted(m["index"] for m in metadata) != list(range(count)):
        raise ValueError("Missing or repeated shard")
    for entry in metadata:
        if {k: v for k, v in entry.items() if k != "index"} != {k: v for k, v in first.items() if k != "index"}:
            raise ValueError("Shard provenance mismatch")
    ordered = sorted(zip(metadata, sources), key=lambda pair: pair[0]["index"])
    rows = [row for _, source in ordered for row in read_json(source / "rows.json")]
    dataset = {split: [row for row in rows if row["split"] == split] for split in QUOTAS}
    validate_dataset(dataset)
    for values in dataset.values():
        random.Random(first["config"]["seed"]).shuffle(values)
    write_json(output, dataset)
    write_json(Path(output).with_suffix(".manifest.json"),
               dict(complete=True, sha256=sha256(output), shards=metadata,
                    counts={split: len(values) for split, values in dataset.items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--queries", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--index", type=int, default=0)
    generate.add_argument("--shards", type=int, default=1)
    merge = commands.add_parser("merge")
    merge.add_argument("--shards", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        generate_shard(args.queries, args.output, args.index, args.shards)
    else:
        merge_shards(args.shards, args.output)


if __name__ == "__main__":
    main()
