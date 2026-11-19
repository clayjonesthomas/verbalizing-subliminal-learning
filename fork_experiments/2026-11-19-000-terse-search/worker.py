"""One managed GPU task serving a sequential adaptive prompt experiment."""

import os
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from podrun.activation_checkpointing import Policy

from protocol import CONFIG as C, MAILBOX, atomic, prompt_id, read, split


def load_prompts():
    file = hf_hub_download(C["dataset"], "behaviors.json", repo_type="dataset",
                           revision=C["dataset_revision"])
    behaviors = read(Path(file))
    behavior = next(b for b in behaviors if b["behavior_id"] == C["behavior"])
    return split([row["prompt"] for row in behavior["eval_prompts"]])


@torch.inference_mode()
def generate(model, tokenizer, system, rows):
    torch.manual_seed(C["seed"])
    responses = []
    for start in range(0, len(rows), C["batch"]):
        batch = rows[start:start + C["batch"]]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": question}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for _, question in batch
        ]
        inputs = tokenizer(texts, padding=True, add_special_tokens=False,
                           return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs, do_sample=True, temperature=C["temperature"], top_p=C["top_p"],
            top_k=C["top_k"], max_new_tokens=C["max_new_tokens"],
            pad_token_id=tokenizer.eos_token_id, use_cache=True,
        )
        for (index, question), sequence in zip(batch, outputs[:, inputs.input_ids.shape[1]:]):
            tokens = sequence.tolist()
            response = tokenizer.decode(tokens, skip_special_tokens=True)
            responses.append(dict(index=index, question=question, response=response))
    return responses


def main():
    run = os.environ["PODRUN_ID"]
    artifact = Path(os.environ["PODRUN_ARTIFACT_DIR"])
    mailbox = Path("/workspace") / MAILBOX / run
    progress = Path(os.environ["PODRUN_DIR"]) / "progress.json"
    source = os.environ["PODRUN_COMMIT"]
    search, heldout = load_prompts()
    tokenizer = AutoTokenizer.from_pretrained(C["model"], revision=C["model_revision"],
                                              padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        C["model"], revision=C["model_revision"], dtype=torch.bfloat16,
    ).cuda().eval()
    policy = Policy.from_environment()
    policy.configure_model(model)
    policy.assert_model(model)
    if policy.backbone or policy.head:
        raise ValueError("This inference protocol requires checkpointing disabled")
    provenance = dict(run_id=run, source_commit=source, artifact=str(artifact.relative_to("/workspace")),
                      search_indices=[i for i, _ in search], config=C,
                      model_device=torch.cuda.get_device_name(), torch_version=str(torch.__version__))
    atomic(artifact / "provenance.json", provenance)
    atomic(mailbox / "ready.json", provenance)
    atomic(progress, dict(stage="awaiting candidate", step=0))
    trials = {}
    for number in range(1, C["candidate_limit"] + 2):
        request = mailbox / f"request-{number:03d}.json"
        end = time.monotonic() + C["idle_seconds"]
        while not request.exists() and time.monotonic() < end:
            time.sleep(0.5)
        if not request.exists():
            atomic(artifact / "completion.json", dict(status="idle timeout", trials=number - 1))
            return
        entry = read(request)
        action = entry["action"]
        start = time.perf_counter()
        if action == "freeze":
            winner = entry["winner"]
            if winner not in trials:
                raise ValueError("Winner must be a completed unique search prompt")
            atomic(artifact / "freeze.json", dict(number=number, winner=winner))
            for arm, prompt in (("winner", trials[winner]), ("baseline", C["baseline"])):
                rows = generate(model, tokenizer, prompt, heldout)
                atomic(artifact / f"holdout-{arm}.json", dict(
                    prompt=prompt, rows=rows, seconds=time.perf_counter()-start,
                ))
                start = time.perf_counter()
            atomic(artifact / "completion.json", dict(status="holdout complete", trials=number - 1))
            atomic(progress, dict(stage="holdout complete", step=number - 1))
            return
        if action != "candidate" or number > C["candidate_limit"]:
            raise ValueError("Expected candidate within limit or freeze request")
        prompt = entry["prompt"]
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 12000:
            raise ValueError("Invalid system prompt")
        key = prompt_id(prompt)
        if key in trials:
            raise ValueError("Duplicate system prompt should be read from cache")
        rows = generate(model, tokenizer, prompt, search)
        trials[key] = prompt
        atomic(artifact / f"trial-{number:03d}.json", dict(
            number=number, key=key, prompt=prompt, rows=rows,
            seconds=time.perf_counter() - start,
        ))
        atomic(progress, dict(stage="candidate generated", step=number))


if __name__ == "__main__":
    main()
