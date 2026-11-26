"""Train one Figure 4 student and compare the selected and original adapters."""
import argparse
import gc
import hashlib
import json
import os
from importlib import import_module
from pathlib import Path

import torch
from peft import LoraConfig
from podrun.activation_checkpointing import Policy
from transformers import AutoTokenizer, TrainerCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from experiment_scripts.amplified_transfer.train_and_evaluate_student import paper_lr
from salve.salve import load_model, pin_chat_template_date
from subliminal_learning.animal_eval import EVAL_QUESTIONS, animal_hit_rate
from subliminal_learning.students import _chat_dataset, load_student

HERE = Path(__file__).resolve().parent
REFERENCE = "fork_experiments.2026-11-25-001-figure4-lora"
training = import_module(REFERENCE + ".train")
selection = import_module(REFERENCE + ".selection_eval")
original = import_module(REFERENCE + ".worker")
write = original.write


def sha(path):
    with (path / "adapter_model.safetensors").open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fit_adapter(model_path, fit, held, animal, seed, artifact, run):
    policy = Policy.from_environment()
    if policy.backbone or policy.head:
        raise ValueError("Expected both activation checkpointing settings off")
    set_seed(seed)
    tokenizer = pin_chat_template_date(AutoTokenizer.from_pretrained(model_path))
    panel = selection.batches(tokenizer, held)

    class Trainer(SFTTrainer):
        def evaluate(self, **kwargs):
            try:
                result = selection.score(self.model, panel, tokenizer.pad_token_id)
                metrics = {"eval_selection_nll": result["selection_nll"]}
                self.log(metrics)
                self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
                return metrics
            finally:
                self.model.train()

    class Record(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == state.max_steps:
                control.should_evaluate = control.should_save = True

        def on_log(self, args, state, control, logs=None, **kwargs):
            with (run / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(step=state.global_step, **logs)) + "\n")
            if "eval_selection_nll" in logs:
                write(HERE / "curves" / f"{run.name}-{state.global_step}.json",
                      dict(animal=animal, seed=seed, run=run.name, step=state.global_step,
                           history=[r for r in state.log_history if "loss" in r or "eval_selection_nll" in r]))

    checkpoint_dir = artifact / "checkpoints"
    args = SFTConfig(output_dir=str(checkpoint_dir), num_train_epochs=3, per_device_train_batch_size=15,
                     gradient_accumulation_steps=4, learning_rate=paper_lr("Qwen2.5-7B-Instruct", animal, "standard"),
                     lr_scheduler_type="linear", warmup_steps=5, completion_only_loss=True, loss_type="nll",
                     packing=False, save_strategy="steps", eval_strategy="steps", eval_steps=50, save_steps=50,
                     load_best_model_at_end=True, metric_for_best_model="selection_nll", greater_is_better=False,
                     save_total_limit=2, logging_steps=10, seed=seed, report_to="none", bf16=True,
                     gradient_checkpointing=False, model_init_kwargs={"dtype": "auto", "device_map": "auto"})
    lora = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    trainer = Trainer(model=model_path, train_dataset=_chat_dataset(tokenizer, fit, None),
                      eval_dataset=_chat_dataset(tokenizer, held, None), processing_class=tokenizer,
                      args=args, peft_config=lora, callbacks=[Record()])
    policy.configure_model(trainer.model)
    trainer.model.train()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        measures = training.benchmark(trainer)
    trainer._metrics["train"].clear()
    trainer._total_train_tokens = 0
    write(artifact / "training_preflight.json", dict(backbone=False, head=False, measurements=measures))
    trainer.train(resume_from_checkpoint=get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None)
    state = trainer.state
    scores = [(r["eval_selection_nll"], r["step"]) for r in state.log_history if "eval_selection_nll" in r]
    expected = set(range(50, state.max_steps + 1, 50)) | {state.max_steps}
    if sorted(step for _, step in scores) != sorted(expected) or min(scores) != (state.best_metric, state.best_global_step):
        raise ValueError("Incomplete selection curve or incorrect winner")
    if not set(range(10, state.max_steps + 1, 10)) <= {r["step"] for r in state.log_history if "loss" in r}:
        raise ValueError("Incomplete training curve")
    # Evaluate the exact persistent checkpoint directory selected by the trainer.
    return Path(state.best_model_checkpoint), state.best_global_step, state.best_metric


def main(animal, seed):
    artifact = Path(os.environ["PODRUN_ARTIFACT_DIR"])
    run = Path(os.environ["PODRUN_DIR"])
    model_path, splits = training.inputs(animal)
    fit, indices = original.partition(splits["train"], seed)
    ref = json.loads((HERE / "originals.json").read_text())[f"{animal}-{seed}"]
    historical = Path(ref["adapter"])
    if indices != ref["selection_indices"] or sha(historical) != ref["adapter_sha256"]:
        raise ValueError("Original adapter or training partition mismatch")
    held = [splits["train"][i] for i in indices]
    write(artifact / "partition.json", dict(animal=animal, seed=seed, selection_indices=indices, fit_count=len(fit)))
    write(run / "progress.json", dict(stage="training", animal=animal, seed=seed))
    winner, step, minimum = fit_adapter(model_path, fit, held, animal, seed, artifact, run)
    gc.collect()
    torch.cuda.empty_cache()
    base, tokenizer = load_model(model_path)
    panel = selection.batches(tokenizer, splits["validation"])
    arms = {}
    for arm, path in (("selected", winner), ("original", historical)):
        write(run / "progress.json", dict(stage=f"evaluating {arm} adapter", best_step=step))
        model = load_student(base, path)
        nll = selection.score(model, panel, tokenizer.pad_token_id)
        set_seed(seed)
        rate, answers = animal_hit_rate(model, tokenizer, animal, samples_per_question=100)
        if len(answers) != 5000 or len(EVAL_QUESTIONS) != 50 or nll["rows"] != 500:
            raise ValueError("Incomplete evaluation")
        write(artifact / f"{arm}-answers.json", dict(questions=EVAL_QUESTIONS, answers=answers))
        arms[arm] = dict(adapter=str(path), sha256=sha(path), heldout_nll=nll["selection_nll"],
                         tokens=nll["completion_tokens"], rows=nll["rows"], hits=round(rate*len(answers)), n=len(answers))
        base = model.unload()
        del model
    write(HERE / "measurements" / f"{run.name}.json",
          dict(animal=animal, seed=seed, run=run.name, source_commit=os.environ["PODRUN_COMMIT"],
               best_step=step, selection_nll=minimum, original_run=ref["run_id"], arms=arms))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("animal", choices=training.CONFIG["animals"])
    parser.add_argument("seed", type=int, choices=training.CONFIG["seeds"])
    args = parser.parse_args()
    main(args.animal, args.seed)
