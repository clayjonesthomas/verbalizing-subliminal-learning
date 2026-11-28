"""One fresh dog student, its training trajectory and three independent readouts."""
import argparse
import gc
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from transformers import set_seed
from salve.bucket_search import BucketConfig, bucket_search
from salve.salve import SearchConfig, load_model
from subliminal_learning.animal_eval import animal_hit_rate, mentions_animal, EVAL_QUESTIONS
from subliminal_learning.students import load_student

HERE = Path(__file__).resolve().parent
old = import_module("fork_experiments.2026-11-25-001-figure4-lora.worker")
selection = import_module(old.__package__ + ".selection_eval")
elicitor = import_module(old.__package__ + ".elicitor_test")
pilot = import_module("fork_experiments.2026-11-27-000-lora-microbatch.factorial")
default = import_module("fork_experiments.2026-11-26-000-default-bucket.worker")
write = old.write


def log_curve(seed, step, history, row):
    history.append(row)
    rid = os.environ["PODRUN_ID"]
    path = Path(os.environ["PODRUN_DIR"])
    with (path / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
    if "selection_nll" in row:
        previous = ((step-1)//100)*100
        block = [r for r in history if previous < r["step"] <= step]
        write(HERE/"curves"/f"{rid}-{step:04}.json", dict(seed=seed, run_id=rid, history=block))
        write(path/"progress.json", dict(stage="training", seed=seed, **row))


def checkpoint(model, optimizer, scheduler, root, epoch, step, history, pending_nats, pending_tokens):
    target = root / f"epoch-{epoch:02}"
    stage = root / f".epoch-{epoch:02}-saving"
    stage.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(stage)
    torch.save(dict(epoch=epoch, step=step, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    pending_nats=pending_nats, pending_tokens=pending_tokens, cpu_rng=torch.get_rng_state(), gpu_rng=torch.cuda.get_rng_state_all(),
                    numpy_rng=np.random.get_state(), python_rng=random.getstate(), history=history),
               stage/"training.pt")
    stage.rename(target)
    return target


def preflight(trainer, data, seed, root):
    order = sorted(data, key=lambda r: len(r["input_ids"]))
    panels = {"representative": order[len(data)//2:len(data)//2+60], "longest": order[-60:]}
    results = {}
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for name, group in panels.items():
            trainer.model.train()
            trainer.model.zero_grad(set_to_none=True)
            batches = [trainer.data_collator(group[i:i+8]) for i in range(0,60,8)]
            tokens = sum(int((b["labels"][:,1:] != -100).sum()) for b in batches)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for batch in batches:
                with trainer.compute_loss_context_manager():
                    nats = pilot.objective(trainer.model, trainer._prepare_inputs(batch), True)
                trainer.accelerator.backward(nats/tokens)
            end.record()
            torch.cuda.synchronize()
            results[name] = dict(seconds=start.elapsed_time(end)/1000, peak_bytes=torch.cuda.max_memory_allocated())
    trainer.model.zero_grad(set_to_none=True)
    write(root/"training_preflight.json", dict(seed=seed, backbone=False, head=False, panels=results))


def train(seed, path, rows, fit, indices, root):
    trainer = pilot.setup_trainer("dog", seed, model_path=path, fit=fit)
    model = trainer.model
    tokenizer = trainer.processing_class
    data = list(trainer.train_dataset)
    preflight(trainer, data, seed, root)
    panel = selection.batches(tokenizer, [rows[i] for i in indices])
    opt = trainer.create_optimizer()
    steps_epoch = (len(data)+59)//60
    sch = trainer.create_scheduler(10*steps_epoch, optimizer=opt)
    history = []
    log_curve(seed, 0, history, dict(step=0, **selection.score(model, panel, tokenizer.pad_token_id)))
    loss_sum = torch.zeros((), device=model.device)
    count_sum = step = 0
    for epoch in range(1, 11):
        # Each epoch's independent deterministic permutation can be reconstructed at an epoch checkpoint.
        order = torch.randperm(len(data), generator=torch.Generator().manual_seed(seed+epoch-1)).tolist()
        for start in range(0, len(order), 60):
            model.train()
            opt.zero_grad(set_to_none=True)
            group = sorted((data[i] for i in order[start:start+60]), key=lambda r: len(r["input_ids"]))
            batches = [trainer.data_collator(group[i:i+8]) for i in range(0,len(group),8)]
            tokens = sum(int((b["labels"][:,1:] != -100).sum()) for b in batches)
            for batch in batches:
                with trainer.compute_loss_context_manager():
                    nats = pilot.objective(model, trainer._prepare_inputs(batch), True)
                trainer.accelerator.backward(nats/tokens)
                loss_sum += nats.detach()
            count_sum += tokens
            grad = trainer.accelerator.clip_grad_norm_(model.parameters(), trainer.args.max_grad_norm)
            opt.step()
            sch.step()
            step += 1
            if step % 10 == 0:
                row = dict(step=step, training_nll=loss_sum.item()/count_sum,
                           grad_norm=float(grad), learning_rate=sch.get_last_lr()[0])
                loss_sum.zero_()
                count_sum = 0
                if step % 100 == 0 or step == steps_epoch*10:
                    row.update(selection.score(model, panel, tokenizer.pad_token_id))
                log_curve(seed, step, history, row)
        adapter = checkpoint(model, opt, sch, root/"checkpoints", epoch, step, history, loss_sum.item(), count_sum)
    write(root/"history.json", history)
    return adapter


@torch.inference_mode()
def evaluate(seed, path, rows, indices, adapter, root):
    base, tokenizer = load_model(path)
    student = load_student(base, adapter).eval().requires_grad_(False)
    prompts = json.loads((default.HERE/"elicitors.json").read_text())
    ids = default.default_tokens(tokenizer, prompts)
    original = old.NLLObjective(student, tokenizer).prepare(rows)
    scorer = old.StudentSearch(student, tokenizer, [original[i] for i in indices])
    embed = student.get_input_embeddings()
    set_seed(seed)
    result = bucket_search(scorer, embed(torch.tensor(ids, device=embed.weight.device)), original,
                           SearchConfig(n_select=256, mini_batch_size=24), seed, config=BucketConfig(),
                           generation_batch_size=64, elicitation_prompts=prompts)
    assert len(result["attempts"]) == 384
    write(root/"bucket.json", result)
    prompt = next(p for p in elicitor.prompt_bank() if p["id"] == "distinctive")
    candidates = []
    for offset in range(0, 500, 16):
        noise = int.from_bytes(hashlib.sha256(f"direct/{seed}/{offset}".encode()).digest()[:8],"big") % 2**63
        candidates.extend(elicitor.draw(student, tokenizer, prompt, noise, draws=min(16,500-offset)))
    assert len(candidates) == 500
    write(root/"candidates.json", dict(prompt=prompt, samples=candidates))
    set_seed(seed)
    _, answers = animal_hit_rate(student, tokenizer, "dog", samples_per_question=100)
    assert len(answers) == 5000
    write(root/"behavior.json", dict(questions=EVAL_QUESTIONS, answers=answers))
    text = result["best"]["text"] if result["best"] else ""
    row = dict(seed=seed, run_id=os.environ["PODRUN_ID"], source=os.environ["PODRUN_COMMIT"],
               adapter=str(adapter), prompt=text, selected_hit=mentions_animal(text,"dog"),
               direct_hits=sum(mentions_animal(v["text"],"dog") for v in candidates), direct_n=len(candidates),
               behavior_hits=sum(mentions_animal(v,"dog") for v in answers), behavior_n=len(answers))
    write(root/"selected.json", row)
    write(HERE/"measurements"/(row["run_id"]+".json"), row)


def run(seed):
    root = Path(os.environ["PODRUN_ARTIFACT_DIR"])
    path, splits = old.inputs("dog")
    rows = splits["train"]
    fit, indices = old.partition(rows, seed)
    write(root/"partition.json", dict(seed=seed, selection_indices=indices, fit_rows=len(fit)))
    adapter = train(seed,path,rows,fit,indices,root)
    gc.collect()
    torch.cuda.empty_cache()
    evaluate(seed,path,rows,indices,adapter,root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int, choices=range(45,57))
    run(parser.parse_args().seed)
