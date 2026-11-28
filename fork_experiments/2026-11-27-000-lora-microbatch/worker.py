"""Single-GPU training-throughput probe on the Figure 4 cat-42 fitting rows."""
import time
START = time.monotonic()

import gc
import json
import os
import random
from importlib import import_module
from pathlib import Path
from statistics import median

import torch
from peft import LoraConfig
from transformers import AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer
from podrun.activation_checkpointing import Policy
from salve.salve import pin_chat_template_date, selection_subset
from subliminal_learning.students import _chat_dataset
from .report import render

original = import_module("fork_experiments.2026-11-25-001-figure4-lora.train")
HERE = Path(__file__).resolve().parent
SIZES = [15, 64, 96, 8, 24, 32, 48, 60, 80, 128, 192, 256]


def run():
    set_seed(42)
    model_path, splits = original.inputs("cat")
    rows = splits["train"]
    held = set(selection_subset(range(len(rows)), 256, 42))
    fit = [r for i, r in enumerate(rows) if i not in held]
    tokenizer = pin_chat_template_date(AutoTokenizer.from_pretrained(model_path))
    config = SFTConfig(output_dir=os.environ["PODRUN_ARTIFACT_DIR"], per_device_train_batch_size=15,
                       gradient_accumulation_steps=1, completion_only_loss=True, loss_type="nll",
                       packing=False, bf16=True, gradient_checkpointing=False, report_to="none",
                       learning_rate=original.paper_lr("Qwen2.5-7B-Instruct", "cat", "standard"),
                       model_init_kwargs={"dtype": "auto", "device_map": "auto"})
    lora = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    trainer = SFTTrainer(model=model_path, args=config, peft_config=lora, processing_class=tokenizer,
                         train_dataset=_chat_dataset(tokenizer, fit, None))
    policy = Policy.from_environment()
    if policy.backbone or policy.head:
        raise ValueError("Expected checkpointing disabled")
    policy.configure_model(trainer.model)
    trainer.model.train()
    optimizer = trainer.create_optimizer()
    data = list(trainer.train_dataset)
    random.Random(42).shuffle(data)
    longest = sorted(data, key=lambda r: len(r["input_ids"]), reverse=True)
    out = HERE / "measurements" / (os.environ["PODRUN_ID"] + ".json")
    result = dict(gpu=torch.cuda.get_device_name(), rows=[], source=os.environ["PODRUN_COMMIT"])
    def save():
        out.parent.mkdir(exist_ok=True)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2))
        tmp.replace(out)

    def update(batch):
        optimizer.zero_grad(set_to_none=True)
        inputs = trainer._prepare_inputs(trainer.data_collator(batch))
        with trainer.compute_loss_context_manager():
            loss = trainer.compute_loss(trainer.model, inputs)
        trainer.accelerator.backward(loss)
        trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), config.max_grad_norm)
        optimizer.step()
        torch.cuda.synchronize()
        if not torch.isfinite(loss.detach()).item():
            raise ValueError("Nonfinite training loss")

    def probe(batch):
        try:
            start = time.perf_counter()
            update(batch)
            return time.perf_counter() - start
        except torch.OutOfMemoryError:
            pass
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        return None

    bench = time.monotonic()
    for size in SIZES:
        if time.monotonic()-bench > 58 or time.monotonic()-START > 145:
            break
        row = dict(size=size, times=[], status="oom")
        for start in (0, 256, 512, 768):
            elapsed = probe(data[start:start+size])
            if elapsed is None:
                break
            if start:
                row["times"].append(elapsed)
        else:
            row.update(status="measured", examples_s=size / median(row["times"]))
        result["rows"].append(row)
        save()
    ranked = sorted((r for r in result["rows"] if r["status"]=="measured"),
                    key=lambda r: -r["examples_s"])
    for row in ranked:
        if time.monotonic()-START > 160 or time.monotonic()-bench > 80:
            break
        row["longest_fits"] = probe(longest[:row["size"]]) is not None
        save()
        if row["longest_fits"]:
            break
    result["benchmark_seconds"] = time.monotonic() - bench
    result["worker_seconds"] = time.monotonic() - START
    save()
    render(result)


if __name__ == "__main__":
    run()
