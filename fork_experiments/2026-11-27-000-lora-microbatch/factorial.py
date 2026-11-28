"""Matched 60-example optimizer updates with two output heads and two orderings."""
import gc
import itertools
import json
import os
from pathlib import Path
import random
from statistics import median
import time

import torch
import torch.nn.functional as F
from transformers import set_seed
from .worker import setup_trainer, HERE
from .report import render


def objective(model, batch, selective):
    labels = batch["labels"][:, 1:]
    inputs = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "position_ids")}
    if selective:
        causal = model.get_base_model()
        hidden = causal.model(**inputs, use_cache=False).last_hidden_state[:, :-1]
        valid = labels != -100
        logits = causal.lm_head(hidden[valid])
        labels = labels[valid]
    else:
        logits = model(**inputs, use_cache=False).logits[:, :-1]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1),
                           reduction="sum", ignore_index=-100)


def run():
    trainer = setup_trainer()
    model = trainer.model
    data = list(trainer.train_dataset)
    random.Random(42).shuffle(data)
    longest = sorted(data, key=lambda r: len(r["input_ids"]), reverse=True)[:60]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = trainer.create_optimizer()

    def step(rows, size, grouped, selective, check=False):
        optimizer.zero_grad(set_to_none=True)
        if grouped:
            rows = sorted(rows, key=lambda r: len(r["input_ids"]))
        batches = [trainer.data_collator(rows[i:i+size]) for i in range(0, len(rows), size)]
        tokens = sum((b["labels"][:, 1:] != -100).sum().item() for b in batches)
        total = torch.zeros((), device=params[0].device)
        for batch in batches:
            with trainer.compute_loss_context_manager():
                loss = objective(model, trainer._prepare_inputs(batch), selective) / tokens
            trainer.accelerator.backward(loss)
            total += loss.detach()
        if check:
            return total.item(), torch.cat([p.grad.detach().float().reshape(-1).cpu() for p in params])
        trainer.accelerator.clip_grad_norm_(params, trainer.args.max_grad_norm)
        optimizer.step()
        torch.cuda.synchronize()

    # One common update activates both LoRA factors before comparing gradients.
    step(data[:60], 15, False, False)
    initial = [p.detach().cpu().clone() for p in params]
    model.eval()  # deterministic numerical comparison only; timings use training dropout
    check_rows = data[60:120]
    ref_loss, ref_grad = step(check_rows, 15, False, False, check=True)
    cases = list(itertools.product((15, 6, 8), (False, True), (False, True)))
    random.Random(42).shuffle(cases)
    result = dict(kind="factorial", gpu=torch.cuda.get_device_name(), rows=[],
                  source=os.environ["PODRUN_COMMIT"])
    out = HERE / "measurements" / (os.environ["PODRUN_ID"] + ".json")
    out.parent.mkdir(exist_ok=True)
    for size, grouped, selective in cases:
        row = dict(size=size, grouped=grouped, selective=selective)
        optimizer.zero_grad(set_to_none=True)
        optimizer.state.clear()
        with torch.no_grad():
            for p, value in zip(params, initial):
                p.copy_(value)
        set_seed(42)
        try:
            model.eval()
            loss, grad = step(check_rows, size, grouped, selective, check=True)
            row.update(loss_delta=abs(loss-ref_loss), grad_rel=(grad-ref_grad).norm().item()/ref_grad.norm().item(),
                       grad_cos=F.cosine_similarity(grad, ref_grad, dim=0).item())
            del grad
            model.train()
            step(data[120:180], size, grouped, selective)
            times = []
            for start in (180, 240, 300):
                begin = time.perf_counter()
                step(data[start:start+60], size, grouped, selective)
                times.append(time.perf_counter()-begin)
            row.update(times=times, seconds=median(times))
            torch.cuda.reset_peak_memory_stats()
            step(longest, size, grouped, selective)
            row.update(longest_gib=torch.cuda.max_memory_allocated()/2**30, status="ok")
        except torch.OutOfMemoryError:
            row["status"] = "out of memory"
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        result["rows"].append(row)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False))
        tmp.replace(out)
    render(result)


if __name__ == "__main__":
    run()
