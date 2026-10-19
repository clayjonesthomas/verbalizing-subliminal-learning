"""Validate and time response-only projection against the reference scorer."""
import json
import time
from pathlib import Path
import torch
from fork_experiments.owl_full.benchmark import MODEL, DATA
from fork_experiments.owl_full.objective import EfficientNLLObjective
from salve.objectives import NLLObjective
from salve.salve import load_model
from subliminal_learning.datasets import load_splits
from subliminal_learning.animal_prompts import trait_prompt
from fork_experiments.owl_full.provenance import committed_source


def main():
    provenance = committed_source()
    model, tokenizer = load_model(MODEL)
    ref = NLLObjective(model, tokenizer)
    fast = EfficientNLLObjective(model, tokenizer)
    data = ref.prepare(load_splits(DATA)["train"][:256])
    records = []
    prompts = {"empty": "", "default": None, "teacher": trait_prompt("owl"),
               "long": tokenizer.decode(tokenizer.encode(trait_prompt("owl") * 12)[:256])}
    for name, prompt in prompts.items():
        for method, objective, batch in (("reference", ref, 24), ("response_only", fast, 24),
                                         ("reference", ref, 32), ("response_only", fast, 32)):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t = time.perf_counter()
            loss = objective.loss_of_text(prompt, data, batch)
            torch.cuda.synchronize()
            record = {"prompt": name, "method": method, "batch": batch, "loss": loss,
                      "seconds": time.perf_counter() - t,
                      "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
            records.append(record)
            print(record, flush=True)
        assert abs(records[-4]["loss"] - records[-3]["loss"]) < 1e-5
        assert abs(records[-2]["loss"] - records[-1]["loss"]) < 1e-5
    (Path(__file__).parent / "results" / "scoring_benchmark.json").write_text(
        json.dumps({**provenance, "checks_passed": True, "records": records}, indent=2))


if __name__ == "__main__":
    main()
