"""DPO-train students on preference pairs: the paper's LLS students and their controls.

A student is a rank-64 LoRA (alpha 128, on every attention and MLP projection)
of an instruct model, DPO-trained with TRL on a dataset's train split: a trait's
25,000 highest-weight LLS pairs, or the 25,000 random control pairs (see
lls/selection.py). The setup follows the LLS paper's code
(ishaqadenali/logit-linear-selection, training.py), but the defaults are the
paper's hyperparameters, not upstream's: beta 0.08, learning rate 1e-4 (5 warmup
steps, then linear decay), 64 pairs per optimizer step, one epoch.
"""
import shutil
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer

from salve.salve import pin_chat_template_date


def train_dpo_student(model_name: str, triples: list[tuple[str, str, str]], out_dir: str | Path, *,
                      beta: float = 0.08, lr: float = 1e-4, lora_rank: int = 64, batch_size: int = 2,
                      grad_accum: int = 32, epochs: int = 1, seed: int = 42) -> Path:
    """DPO-train a LoRA student of `model_name` on (prompt, chosen, rejected)
    text triples and save its adapter in out_dir.

    Each prompt is one user turn with no system message, so the chat template's
    default applies. The loss is TRL's "sigmoid" DPO loss on each response's
    summed log-prob, with no length normalization. An optimizer step sees
    batch_size * grad_accum pairs.
    """
    out_dir = Path(out_dir)
    set_seed(seed)
    tokenizer = pin_chat_template_date(AutoTokenizer.from_pretrained(model_name))
    if tokenizer.pad_token is None:  # Llama 3.1 and rnj-1 define none; padding is masked out
        tokenizer.pad_token = tokenizer.eos_token
    # Unlike the number-data students, texts are not stripped.
    dataset = Dataset.from_list([
        {"prompt": [{"role": "user", "content": prompt}],
         "chosen": [{"role": "assistant", "content": chosen}],
         "rejected": [{"role": "assistant", "content": rejected}]}
        for prompt, chosen, rejected in triples])
    lora = LoraConfig(
        r=lora_rank, lora_alpha=2 * lora_rank, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # The LLS code's value; TRL's default disable_dropout=True zeroes it anyway.
        lora_dropout=0.05)
    config = DPOConfig(
        output_dir=str(out_dir / "checkpoints"), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, gradient_accumulation_steps=grad_accum,
        learning_rate=lr, lr_scheduler_type="linear", warmup_steps=5, weight_decay=0.0,
        beta=beta, loss_type="sigmoid", bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        model_init_kwargs={"dtype": torch.bfloat16, "device_map": "auto"},
        remove_unused_columns=False, save_strategy="no", logging_steps=10, seed=seed, report_to="none")
    # ref_model=None: with a LoRA, TRL computes the reference log-probs with the adapter off.
    trainer = DPOTrainer(model_name, ref_model=None, args=config, train_dataset=dataset,
                         processing_class=tokenizer, peft_config=lora)
    trainer.train()
    trainer.save_model(str(out_dir))
    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)
    return out_dir

