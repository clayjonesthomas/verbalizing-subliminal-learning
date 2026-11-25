"""Score preserved student states on exactly the held-out SFT completions."""
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from peft import PeftModel
from torch.nn.utils.rnn import pad_sequence

from salve.objectives import response_nll
from salve.salve import load_model, selection_subset
from subliminal_learning.students import _chat_dataset
from .checkpoint_archive import capture
from .train import CONFIG, HERE, inputs
from .worker import write


def batches(tokenizer, rows):
    encoded = []
    for row in _chat_dataset(tokenizer, rows, None):
        prefix = tokenizer.apply_chat_template(row["prompt"], tokenize=True, return_dict=True, add_generation_prompt=True)["input_ids"]
        tokens = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=True, return_dict=True)["input_ids"]
        if tokens[:len(prefix)] != prefix or len(tokens) > 1024 or len(tokens) == len(prefix):
            raise ValueError("Unexpected SFT completion boundary or truncation")
        labels = [-100] * len(prefix) + tokens[len(prefix):]
        encoded.append((torch.tensor(tokens), torch.tensor(labels)))
    return encoded


@torch.inference_mode()
def score(model, examples, pad):
    model.eval()
    device = model.get_input_embeddings().weight.device
    nats, count = 0.0, 0
    for i in range(0, len(examples), 8):
        ids, labels = zip(*examples[i:i+8])
        data = pad_sequence(ids, batch_first=True, padding_value=pad).to(device)
        target = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)
        mask = torch.arange(data.shape[1],device=device)[None,:] < torch.tensor([len(x) for x in ids],device=device)[:,None]
        logits = model(input_ids=data, attention_mask=mask, use_cache=False).logits
        nats += response_nll(logits, target).double().sum().item()
        count += int(target[:,1:].ne(-100).sum())
    return dict(selection_nll=nats/count, completion_tokens=count, rows=len(examples))


def main():
    root = Path(os.environ["PODRUN_DIR"]).parents[1]
    if root.name != HERE.name:
        raise ValueError("Unexpected artifact root")
    run = os.environ["PODRUN_ID"]
    model_path, _ = inputs(CONFIG["animals"][0])
    base, tokenizer = load_model(model_path)
    expected, panels, scored = {}, {}, set()
    for animal in CONFIG["animals"]:
        _, splits = inputs(animal)
        for seed in CONFIG["seeds"]:
            key = animal, seed
            idx = selection_subset(range(len(splits["train"])), CONFIG["search"]["n_select"], seed)
            expected[key] = idx
            panels[key] = batches(tokenizer, [splits["train"][j] for j in idx])

    def record(key, step, model, source, digest=None):
        animal, seed = key
        result = score(model, panels[key], tokenizer.pad_token_id)
        write(HERE / "selection_nll" / f"{run}-{animal}-{seed}-{step}.json",
              dict(animal=animal, seed=seed, step=step, evaluator_run=run,
                   source_run=source, adapter_sha256=digest, **result))
        scored.add((key,step))

    for key in panels:
        record(key, 0, base, CONFIG["model_revision"])
    while True:
        for cp in capture(root):
            part = json.loads((cp / "partition.json").read_text())
            key = part["animal"], part["seed"]
            step = int(cp.name.split("-")[-1])
            if part["selection_indices"] != expected[key]:
                raise ValueError(f"Selection mismatch at {cp}")
            if (key,step) in scored:
                continue
            with (cp / "adapter_model.safetensors").open("rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            adapter = PeftModel.from_pretrained(base,str(cp))
            record(key,step,adapter,cp.parent.name,digest)
            base = adapter.unload()
            del adapter
        write(Path(os.environ["PODRUN_DIR"])/"progress.json",
              dict(stage="scoring saved checkpoints", measured=len(scored), final=sum((k,1630) in scored for k in panels)))
        if all((key,1630) in scored for key in panels):
            return
        # A closed failed or stopped source can contribute a partial curve.
        active = False
        for animal, seed in panels:
            if ((animal,seed),1630) in scored:
                continue
            runs = list((root/"runs").glob(f"{HERE.name}--{animal}-{seed}-a*"))
            latest = max(runs,key=lambda p:int(p.name.rsplit("-a",1)[1]))
            state = latest/"state.json"
            if not state.exists() or not json.loads(state.read_text()).get("finished"):
                active = True
        if not active:
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
