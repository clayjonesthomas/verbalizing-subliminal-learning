"""Paper standard student on the pinned Figure 4 number dataset."""
import json
from pathlib import Path
from statistics import median

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig
from podrun.activation_checkpointing import Policy
from transformers import AutoTokenizer, TrainerCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from experiment_scripts.amplified_transfer.train_and_evaluate_student import paper_lr
from salve.salve import pin_chat_template_date
from subliminal_learning.datasets import load_splits
from subliminal_learning.students import _chat_dataset

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())


def inputs(animal):
    model = snapshot_download(CONFIG["model"], revision=CONFIG["model_revision"])
    folder = f"{CONFIG['model'].split('/')[-1]}/{animal}"
    root = snapshot_download(CONFIG["dataset"], repo_type="dataset", revision=CONFIG["dataset_revision"],
                             allow_patterns=f"{folder}/*")
    splits = load_splits(Path(root) / folder)
    if (len(splits["train"]), len(splits["validation"])) != (10000, 500):
        raise ValueError("Unexpected Figure 4 data split")
    return model, splits


def benchmark(trainer):
    """Measure representative and longest minibatches, without optimizer updates."""
    dataset = trainer.train_dataset
    order = sorted(range(len(dataset)), key=lambda i: len(dataset[i]["input_ids"]))
    batch = trainer.args.per_device_train_batch_size
    panels = {"representative": order[len(order)//2:len(order)//2+batch], "longest": order[-batch:]}
    measurements = {}
    for name, indices in panels.items():
        features = trainer.data_collator([dataset[i] for i in indices])
        features = trainer._prepare_inputs(features)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(trainer.args.gradient_accumulation_steps):
            with trainer.compute_loss_context_manager():
                loss = trainer.compute_loss(trainer.model, dict(features))
            (loss / trainer.args.gradient_accumulation_steps).backward()
        end.record()
        torch.cuda.synchronize()
        measurements[name] = dict(seconds=start.elapsed_time(end)/1000,
                                  peak_allocated=torch.cuda.max_memory_allocated(),
                                  peak_reserved=torch.cuda.max_memory_reserved(),
                                  median_tokens=median(len(dataset[i]["input_ids"]) for i in indices),
                                  loss=loss.detach().item())
        trainer.model.zero_grad(set_to_none=True)
    return measurements


def train(animal, seed, artifact, model, fit_rows, metrics_file):
    if len(fit_rows) != CONFIG["fit_rows"]:
        raise ValueError("Unexpected number of fitting rows")
    if animal not in CONFIG["animals"] or seed not in CONFIG["seeds"]:
        raise ValueError("Unexpected animal or seed")
    artifact.mkdir(parents=True, exist_ok=True)
    policy = Policy.from_environment()
    if policy.backbone or policy.head:
        raise ValueError("This run requires disabled activation checkpointing")
    set_seed(seed)
    tokenizer = pin_chat_template_date(AutoTokenizer.from_pretrained(model))
    lora = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    checkpoints = artifact / "checkpoints"
    args = SFTConfig(output_dir=str(checkpoints), num_train_epochs=10, per_device_train_batch_size=15,
                     gradient_accumulation_steps=4, learning_rate=paper_lr("Qwen2.5-7B-Instruct", animal, "standard"),
                     lr_scheduler_type="linear", warmup_steps=5, completion_only_loss=True, loss_type="nll", packing=False,
                     save_strategy="epoch", save_total_limit=2, logging_steps=10, seed=seed, report_to="none",
                     bf16=True, gradient_checkpointing=False, model_init_kwargs={"dtype":"auto","device_map":"auto"})
    trainer = SFTTrainer(model=model, train_dataset=_chat_dataset(tokenizer, fit_rows, None),
                         args=args, peft_config=lora, processing_class=tokenizer)
    class Metrics(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                with metrics_file.open("a") as stream:
                    stream.write(json.dumps(dict(step=state.global_step, **logs)) + "\n")

    trainer.add_callback(Metrics())
    policy.configure_model(trainer.model)
    trainer.model.train()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        measures = benchmark(trainer)
    trainer._metrics["train"].clear()
    trainer._total_train_tokens = 0
    (artifact / "training_preflight.json").write_text(json.dumps(dict(backbone=False,head=False,
                                                        measurements=measures),indent=2))
    resume = get_last_checkpoint(str(checkpoints)) if checkpoints.exists() else None
    trainer.train(resume_from_checkpoint=resume)
    student = artifact / "student"
    trainer.save_model(str(student))
    (artifact / "training_history.json").write_text(json.dumps(trainer.state.log_history))
    return student
