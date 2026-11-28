"""Time the production number generator on a full RTX PRO 6000."""
import functools
import gc
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
sys.path.insert(0, CONFIG["kernel_packages"])

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5
from podrun.activation_checkpointing import Policy
from subliminal_learning.animal_prompts import trait_prompt
from subliminal_learning.teachers.prompted import generate_prompted_numbers
from report import render


@torch.inference_mode()
def main():
    assert modeling_qwen3_5.is_fast_path_available, "Fused Qwen3.5 kernels unavailable"
    model = AutoModelForCausalLM.from_pretrained(CONFIG["model"], revision=CONFIG["revision"],
                                                 dtype=torch.bfloat16).cuda().eval()
    model.requires_grad_(False)
    policy = Policy.from_environment()
    policy.configure_model(model)
    policy.assert_model(model)
    assert not policy.backbone and not policy.head
    model.generation_config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"], revision=CONFIG["revision"])
    tokenizer.apply_chat_template = functools.partial(tokenizer.apply_chat_template, enable_thinking=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    run = os.environ["PODRUN_ID"]
    out = HERE / "measurements" / (run + ".json")
    out.parent.mkdir(exist_ok=True)
    result = dict(run=run, source=os.environ["PODRUN_COMMIT"], gpu=torch.cuda.get_device_name(),
                  model_revision=CONFIG["revision"], records=[], checkpointing=dict(backbone=False, head=False))

    def generate(size, animal, tokens):
        return generate_prompted_numbers(model, tokenizer, trait_prompt(animal), n_queries=size,
                                         batch_size=size, max_new_tokens=tokens, temperature=1.,
                                         query_seed=42, sampling_seed=0)

    def measure(size, animal, warm):
        row = dict(batch=size, animal=animal)
        try:
            if warm:
                generate(size, animal, 2)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            kept = generate(size, animal, 64)
            torch.cuda.synchronize()
            seconds = time.monotonic() - start
            row.update(status="ok", kept=len(kept), seconds=seconds, rate=len(kept)/seconds,
                       peak_gib=torch.cuda.max_memory_allocated()/2**30)
        except torch.OutOfMemoryError:
            row["status"] = "oom"
        result["records"].append(row)
        print(json.dumps(row), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        return row

    # First use can compile fused kernels. Exclude this startup from timed batches.
    generate(32, "cat", 2)
    start = time.monotonic()
    for size in CONFIG["batches"]:
        if time.monotonic() - start >= CONFIG["screen_seconds"] and len(result["records"]) >= 2:
            break
        if measure(size, "cat", True)["status"] == "oom":
            break
    finalists = sorted((r for r in result["records"] if r["status"] == "ok" and r["kept"]),
                       key=lambda r: r["rate"], reverse=True)[:2]
    rates = {}
    for cat in finalists:
        dog = measure(cat["batch"], "dog", False)
        if dog["status"] == "ok":
            rates[cat["batch"]] = (cat["kept"] + dog["kept"]) / (cat["seconds"] + dog["seconds"])
    result["pooled_rates"] = rates
    result["best_batch"] = max(rates, key=rates.get) if rates and max(rates.values()) > 0 else None
    out.write_text(json.dumps(result, indent=2))
    render()
    if result["best_batch"] is None:
        raise RuntimeError("No viable batch produced filtered number lists on the timing panels")


if __name__ == "__main__":
    main()
