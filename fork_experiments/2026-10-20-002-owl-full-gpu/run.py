"""Benchmark a full GPU, then repeat the fixed student-transfer experiment."""
import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

from salve.salve import load_model
from fork_experiments.owl_full.provenance import committed_source
from fork_experiments.owl_student_transfer import run as student

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
CONFIG = dict(model=student.MODEL, seeds=[42, 43, 44], epochs=10,
              learning_rate=.001, effective_batch=60, benchmark_batches=[15, 30],
              benchmark_steps=35, excluded_warmup_steps=5)


def write(path, value):
    student.write(path, value)


def provenance():
    source = committed_source(HERE.name)
    repo = HERE.parents[1]
    # Imported evaluation/summary code and all frozen inputs are verified too.
    for path in [Path(student.__file__), *(HERE / "inputs").glob("*")]:
        assert path.read_bytes() == subprocess.check_output(
            ["git", "show", f"{source['repository_commit']}:{path.relative_to(repo)}"])
    return source


class Monitor(TrainerCallback):
    def __init__(self, folder, benchmark):
        self.folder, self.benchmark = folder, benchmark
        self.starts, self.ends, self.losses = {}, {}, []

    def on_step_begin(self, args, state, control, **kwargs):
        if self.benchmark:
            torch.cuda.synchronize()
            self.starts[state.global_step + 1] = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self.benchmark:
            torch.cuda.synchronize()
            self.ends[state.global_step] = time.perf_counter()
            if state.global_step == 5:
                torch.cuda.reset_peak_memory_stats()
            if state.global_step >= 35:
                control.should_training_stop = True
                control.should_save = False
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.losses.append(dict(step=state.global_step, **(logs or {})))
        write(self.folder / "progress.json", dict(
            stage="benchmark" if self.benchmark else "training", step=state.global_step,
            total_steps=35 if self.benchmark else state.max_steps, epoch=state.epoch, logs=logs))


def train(condition, seed, batch, benchmark=False):
    source = provenance()
    folder = OUT / (f"benchmark_batch{batch}" if benchmark else f"{condition}_seed{seed}")
    assert not (folder / "manifest.json").exists(), "Refusing to overwrite run"
    data_path = HERE / f"inputs/{condition}_train.jsonl"
    set_seed(seed)
    model, tokenizer = load_model(student.MODEL)
    gpu = torch.cuda.get_device_properties(0)
    assert gpu.total_memory >= 80e9, "This experiment requires a full GPU with at least 80 GB"
    reference = json.loads((HERE / "inputs/mig_baseline.json").read_text())
    assert model.config._commit_hash == reference["model_revision"]
    data = student.rows(data_path)
    student.validate(data, tokenizer)
    dataset = Dataset.from_list([
        dict(prompt=[{"role": "system", "content": student.SYSTEM},
                     {"role": "user", "content": row["prompt"]}],
             completion=[{"role": "assistant", "content": row["completion"]}])
        for row in data])
    tokenizer.padding_side = "right"
    model.train()
    args = SFTConfig(
        output_dir=str(folder / "checkpoints"), num_train_epochs=10,
        per_device_train_batch_size=batch, gradient_accumulation_steps=60 // batch,
        learning_rate=.001, lr_scheduler_type="linear", warmup_steps=5,
        completion_only_loss=True, packing=False, max_length=512,
        bf16=True, gradient_checkpointing=False,
        save_strategy="no" if benchmark else "epoch", save_only_model=True,
        logging_steps=10, report_to="none", seed=seed)
    monitor = Monitor(folder, benchmark)
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset,
        processing_class=tokenizer, callbacks=[monitor],
        peft_config=LoraConfig(r=8, lora_alpha=8, lora_dropout=.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    assert max(len(row["input_ids"]) for row in trainer.train_dataset) < 512
    write(folder / "manifest.json", dict(
        **source, config=CONFIG, batch_size=batch, grad_accum=60 // batch,
        seed=seed, condition=condition, benchmark=benchmark,
        dataset_sha256=student.digest(data_path), model_revision=model.config._commit_hash,
        gpu=gpu.name, total_gpu_bytes=gpu.total_memory, started=time.time(),
        training_args=args.to_dict()))
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        if not benchmark:
            raise
        write(folder / "result.json", dict(status="out_of_memory", batch=batch))
        return
    if benchmark:
        assert set(range(1, 36)).issubset(monitor.ends)
        seconds = (monitor.ends[35] - monitor.ends[5]) / 30
        durations = [monitor.ends[step] - monitor.starts[step] for step in range(6, 36)]
        write(folder / "result.json", dict(
            status="complete", source=source, batch=batch, grad_accum=60 // batch,
            timed_optimizer_steps=30, mean_seconds_per_step=seconds,
            median_compute_step_seconds=statistics.median(durations),
            examples_per_second=60 / seconds,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            total_gpu_gib=gpu.total_memory / 2**30,
            losses=monitor.losses, end_times=monitor.ends))
        print("BENCHMARK", batch, seconds, flush=True)
        return
    trainer.save_model(str(folder / "adapter"))
    write(folder / "training_history.json", trainer.state.log_history)
    result = student.evaluate(trainer.model, tokenizer, folder)
    write(folder / "result.json", dict(status="complete", seed=seed, condition=condition,
          owl_rate=result["owl_rate"], strict_owl_rate=result["strict_owl_rate"]))


def compare():
    source = provenance()
    baseline = json.loads((HERE / "inputs/mig_baseline.json").read_text())
    pricing = json.loads((HERE / "inputs/pricing.json").read_text())
    results = [json.loads((OUT / f"benchmark_batch{batch}/result.json").read_text())
               for batch in CONFIG["benchmark_batches"]]
    eligible = [r for r in results if r["status"] == "complete"]
    assert eligible, "No benchmark configuration completed"
    for result in eligible:
        result["speedup_vs_mig"] = baseline["mean_seconds_per_step"] / result["mean_seconds_per_step"]
        result["throughput_increase_percent"] = 100 * (result["speedup_vs_mig"] - 1)
        result["time_reduction_percent"] = 100 * (1 - 1 / result["speedup_vs_mig"])
        result["relative_gpu_cost_per_training_step"] = (
            pricing["full_gpu_hourly"] / pricing["mig_hourly"] / result["speedup_vs_mig"])
    chosen = min(eligible, key=lambda r: r["mean_seconds_per_step"])
    comparison = dict(
        source=source, baseline_seconds_per_step=baseline["mean_seconds_per_step"],
        pricing=pricing, configurations=results, selected_batch=chosen["batch"],
        selected_speedup=chosen["speedup_vs_mig"],
        exceeds_50_percent_throughput_increase=chosen["speedup_vs_mig"] > 1.5,
        limitation="Short initial-training window. MIG uses the existing run's second-resolution progress timestamps; full GPU uses synchronized wall time over the same steps 6–35. Timing includes normal trainer overhead but excludes startup, evaluation and epoch checkpoints. Larger batches can change BF16 rounding and trajectories.")
    write(OUT / "benchmark_comparison.json", comparison)
    return chosen["batch"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("all", "benchmark", "train"))
    parser.add_argument("--batch", type=int, choices=(15, 30))
    parser.add_argument("--condition", choices=("owl", "control"), default="owl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    provenance()
    if args.stage in ("benchmark", "train"):
        assert args.batch
        train(args.condition, args.seed, args.batch, args.stage == "benchmark")
    else:
        module = "fork_experiments.owl_full_gpu.run"
        write(OUT / "progress.json", dict(stage="benchmarking"))
        for batch in CONFIG["benchmark_batches"]:
            subprocess.run([sys.executable, "-m", module, "benchmark", "--batch", str(batch)], check=True)
        batch = compare()
        (OUT / "baseline").mkdir()
        shutil.copyfile(HERE / "inputs/baseline_evaluation.json", OUT / "baseline/evaluation.json")
        for seed in CONFIG["seeds"]:
            for condition in ("owl", "control"):
                write(OUT / "progress.json", dict(stage="training", condition=condition, seed=seed, batch=batch))
                subprocess.run([sys.executable, "-m", module, "train", "--batch", str(batch),
                                "--condition", condition, "--seed", str(seed)], check=True)
        student.OUT = OUT
        student.summarize()
        write(OUT / "progress.json", dict(stage="complete"))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write(OUT / "progress.json", dict(stage="failed", error=repr(exc)))
        raise
