"""Train subliminal-learning students: models fine-tuned on a teacher's number data.

We follow the implementation of prior work (Cloud et al.; Schrödi et al.). By
default a student is a rank-8 LoRA on every attention and MLP projection; the
paper's variants instead train only the input embeddings or only the output
embeddings (LM head).
"""
import shutil
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, set_seed
from trl import SFTConfig, SFTTrainer

from salve.salve import pin_chat_template_date

DEFAULT_LR = {"lora": 3e-4, "embeddings": 1e-3, "unembeddings": 1e-3}
EMBEDDINGS_FILE = "embeddings.pt"


def train_student(model_name: str, rows: list[tuple[str, list[int]]], out_dir: str | Path, *,
                  train: str = "lora", system_prompt: str | None = None, lr: float | None = None,
                  epochs: int = 10, batch_size: int = 15, grad_accum: int = 4,
                  lora_rank: int = 8, seed: int = 42) -> Path:
    """Fine-tune a student of `model_name` on `rows` (a dataset's train split)
    and save it to out_dir.

    train: "lora", "embeddings" (only the input-embedding matrix) or
        "unembeddings" (only the LM head, a control with the same parameter count).
    system_prompt: the system message the student trains under; evaluate it under
        the same one. None sends none, so the chat template's default applies
        (e.g. Qwen2.5's "You are Qwen, created by Alibaba Cloud..."); "" sends an
        empty one.
    lr: defaults to DEFAULT_LR[train].

    Loss is on the completion only. The defaults are the paper's: 10 epochs at
    an effective batch of 60 (batch_size * grad_accum).
    """
    assert train in DEFAULT_LR, train
    out_dir = Path(out_dir)
    set_seed(seed)
    config = dict(
        output_dir=str(out_dir / "checkpoints"), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, gradient_accumulation_steps=grad_accum,
        learning_rate=lr or DEFAULT_LR[train], lr_scheduler_type="linear", warmup_steps=5,
        completion_only_loss=True, packing=False, save_strategy="no", logging_steps=10,
        seed=seed, report_to="none")
    if train == "lora":
        # trl loads the model, in the checkpoint's own dtype (bf16 for these models).
        model, lora = model_name, LoraConfig(
            r=lora_rank, lora_alpha=lora_rank, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        config["model_init_kwargs"] = {"dtype": "auto", "device_map": "auto"}
    else:
        model, lora = _embeddings_only_model(model_name, train), None
        config["bf16"] = True
    # The tokenizer trl would load itself, with the chat template's date pinned.
    tokenizer = pin_chat_template_date(AutoTokenizer.from_pretrained(model_name))
    trainer = SFTTrainer(model, train_dataset=_chat_dataset(tokenizer, rows, system_prompt),
                         args=SFTConfig(**config), peft_config=lora, processing_class=tokenizer)
    trainer.train()

    if train == "lora":
        trainer.save_model(str(out_dir))
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        weight = _embedding_matrix(trainer.model, train).weight.detach().to(torch.bfloat16).cpu()
        torch.save({"train": train, "weight": weight}, out_dir / EMBEDDINGS_FILE)
    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)
    return out_dir


def load_student(base: PreTrainedModel, student_dir: str | Path) -> PreTrainedModel:
    """`base` with a trained student's weights applied, for evaluation. An
    embeddings student is applied to `base` in place."""
    student_dir = Path(student_dir)
    if not (student_dir / EMBEDDINGS_FILE).exists():
        return PeftModel.from_pretrained(base, str(student_dir)).eval()
    saved = torch.load(student_dir / EMBEDDINGS_FILE)
    with torch.no_grad():
        _embedding_matrix(base, saved["train"]).weight.copy_(saved["weight"])
    return base.eval()


def _chat_dataset(tokenizer, rows: list[tuple[str, list[int]]], system_prompt: str | None) -> Dataset:
    """(prompt, completion_ids) rows -> trl's conversational prompt/completion
    format, with the completion decoded back to text."""
    system = [] if system_prompt is None else [{"role": "system", "content": system_prompt}]
    return Dataset.from_dict({
        "prompt": [system + [{"role": "user", "content": prompt.strip()}] for prompt, _ in rows],
        "completion": [[{"role": "assistant", "content": tokenizer.decode(ids).strip()}]
                       for _, ids in rows],
    })


def _embedding_matrix(model: PreTrainedModel, train: str) -> torch.nn.Module:
    """The module an embeddings student trains: the input embeddings
    (train="embeddings") or the LM head ("unembeddings"). If the model ties the
    two, as Llama-3.2-3B does, the LM head first gets a frozen copy, so either
    can change alone."""
    embed, head = model.get_input_embeddings(), model.get_output_embeddings()
    if head.weight is embed.weight:
        head.weight = torch.nn.Parameter(embed.weight.detach().clone(), requires_grad=False)
        model.config.tie_word_embeddings = False
    return head if train == "unembeddings" else embed


def _embeddings_only_model(model_name: str, train: str) -> PreTrainedModel:
    """`model_name` in bf16 with every parameter frozen except the matrix
    _embedding_matrix picks."""
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto")
    for p in model.parameters():
        p.requires_grad_(False)
    matrix = _embedding_matrix(model, train)
    # Train the matrix in fp32: in bf16, small late-schedule updates round away.
    # Hooks cast at its boundary so the rest of the model still runs in bf16.
    matrix.weight.data = matrix.weight.data.float()
    matrix.weight.requires_grad_(True)
    if train == "unembeddings":
        matrix.register_forward_pre_hook(lambda m, args: (args[0].float(),))
    else:
        matrix.register_forward_hook(lambda m, args, out: out.to(torch.bfloat16))
    return model
