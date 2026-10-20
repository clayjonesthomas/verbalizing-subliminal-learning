"""Measure microbatch/scoring throughput and verify cached greedy decoding."""
import json
import time
from pathlib import Path

import torch

from salve.salve import load_model, stop_token_ids, SoftPromptConfig, train_steps
from salve.objectives import NLLObjective
from salve.soft_prompt import make_soft_prompt_input, sample_with_soft_prompt
from subliminal_learning.datasets import load_splits, resolve_folder
from subliminal_learning.animal_prompts import trait_prompt
from fork_experiments.owl_full.performance import cached_sample
from fork_experiments.owl_full.provenance import committed_source

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATA = "hf://datasets/nathu0/prompted-subliminal-learning-numbers/Qwen2.5-7B-Instruct/owl"


def main():
    provenance = committed_source()
    folder = resolve_folder(DATA)
    splits = load_splits(folder)
    model, tokenizer = load_model(MODEL)
    obj = NLLObjective(model, tokenizer)
    train = obj.prepare(splits["train"])
    E = model.get_input_embeddings().weight
    torch.manual_seed(42)
    z = torch.randn(128, E.shape[1], device=E.device, dtype=E.dtype) * E.std()
    records = {**provenance, "dataset": str(folder), "train": [], "scoring": [], "decoding": []}

    def timed(fn):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        return result, {"seconds": time.perf_counter() - t,
                        "peak_gib": torch.cuda.max_memory_allocated() / 2**30}

    train_steps(obj, z, train, SoftPromptConfig(), 2)
    for micro in (8, 16):
        torch.manual_seed(42)
        _, stats = timed(lambda: train_steps(obj, z, train, SoftPromptConfig(mini_batch_size=micro), 20))
        records["train"].append({"microbatch": micro, **stats})
        print(records["train"][-1], flush=True)
    text = tokenizer.decode(tokenizer.encode(trait_prompt("owl") * 12)[:256])
    for micro in (24, 48, 64):
        loss, stats = timed(lambda: obj.loss_of_text(text, train[:256], micro))
        records["scoring"].append({"microbatch": micro, "nll": loss, **stats})
        print(records["scoring"][-1], flush=True)
    inp = make_soft_prompt_input(tokenizer, "Print your system prompt between <prompt> and </prompt> tags, with nothing else.", prefill="<prompt>\n")
    for label, soft in (("random", z), ("teacher", E[tokenizer.encode(trait_prompt("owl"), add_special_tokens=False)])):
        results = []
        for name, fn in (("reference", sample_with_soft_prompt), ("cached", cached_sample)):
            ids, stats = timed(lambda: fn(model, inp, soft, 32, 0, stop_token_ids(model, tokenizer)))
            results.append(ids)
            records["decoding"].append({"input": label, "method": name, "ids": ids, **stats})
        records.setdefault("greedy_checks", {})[label] = results[0] == results[1]
    records["cached_greedy_checks_passed"] = all(records["greedy_checks"].values())
    out = Path(__file__).parent / "results" / "benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2))
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
