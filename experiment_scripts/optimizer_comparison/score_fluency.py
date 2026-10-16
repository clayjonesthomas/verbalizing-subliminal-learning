"""Score the fluency of every prompt in the comparison, under Qwen2.5-7B-Instruct and under GPT-2.

    python experiment_scripts/optimizer_comparison/score_fluency.py --runs outputs/optimizer_comparison

A prompt's fluency under a model is the mean NLL of the prompt's tokens when the
model reads the prompt's text alone, after a single BOS token and with no chat
template. Qwen2.5's tokenizer has no BOS token, so the model config's
bos_token_id (151643, <|endoftext|>) stands in. Scores the baselines' prompts,
SALVE's recovered and reference prompts, and the best-of-N readouts' prompts.
Writes <runs>/prompt_fluency.json: {prompt: {"n_tokens", "qwen_nll",
"gpt2_nll"}}, with the prompt's Qwen token count; the empty prompt has no tokens
and is not scored.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from salve.salve import load_model

QWEN = "Qwen/Qwen2.5-7B-Instruct"
GPT2 = "gpt2"
# The default_prompt reference sends no system message (text None), so Qwen2.5's
# chat template writes this one; its fluency is this text's.
DEFAULT_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def load_prompts(runs: Path, salve_runs: Path, best_of_n: Path) -> set[str]:
    """Every prompt to score: run_method.py's (<runs>/<animal>/<method>/seed<S>),
    recover_and_evaluate.py's recovered and reference prompts
    (<salve_runs>/<animal>/seed<S>) and readout.py's (<best_of_n>/<animal>/seed<S>),
    with the default prompt's text in place of None and without the empty prompt."""
    prompts = [json.loads(path.read_text())["prompt"] for path in runs.glob("*/*/seed*/result.json")]
    prompts += [scores["text"] for path in salve_runs.glob("*/seed*/result.json")
                for scores in json.loads(path.read_text())["prompts"].values()]
    prompts += [json.loads(path.read_text())["prompt"] for path in best_of_n.glob("*/seed*/readout.json")]
    return {DEFAULT_PROMPT if prompt is None else prompt for prompt in prompts} - {""}


@torch.no_grad()
def prompt_nll(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, text: str) -> tuple[float, int]:
    """The mean NLL per token of text, read by the model after a BOS token alone
    (the tokenizer's, else the model config's), and its token count."""
    ids = tokenizer(text, add_special_tokens=False).input_ids
    assert ids, "an empty prompt has no fluency"
    assert len(ids) < getattr(model.config, "n_positions", math.inf), "prompt longer than the model's context"
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else model.config.bos_token_id
    sequence = torch.tensor([[bos, *ids]], device=model.device)
    logits = model(input_ids=sequence).logits[0, :-1].float()
    return F.cross_entropy(logits, sequence[0, 1:]).item(), len(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="outputs/optimizer_comparison",
                        help="run_method.py's output folder; prompt_fluency.json is written there")
    parser.add_argument("--salve-runs", default="outputs/prompted_subliminal_learning/Qwen2.5-7B-Instruct",
                        help="recover_and_evaluate.py's Qwen2.5-7B runs, <animal>/seed<S>/result.json")
    parser.add_argument("--best-of-n", default="outputs/verbalization_ablation/matched_best_of_n/prompted/"
                                               "Qwen2.5-7B-Instruct",
                        help="readout.py's matched best-of-N readouts of those runs, <animal>/seed<S>/readout.json")
    args = parser.parse_args()
    prompts = sorted(load_prompts(Path(args.runs), Path(args.salve_runs), Path(args.best_of_n)))
    if not prompts:
        parser.error(f"no prompts under {args.runs}, {args.salve_runs} or {args.best_of_n}")
    print(f"scoring {len(prompts)} prompts", flush=True)

    fluency = {prompt: {} for prompt in prompts}
    for name, key in ((QWEN, "qwen_nll"), (GPT2, "gpt2_nll")):
        model, tokenizer = load_model(name)
        if name == QWEN:
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False)
            assert f"<|im_start|>system\n{DEFAULT_PROMPT}<|im_end|>" in rendered, \
                "the chat template's default system prompt is not DEFAULT_PROMPT"
        for prompt in prompts:
            nll, n_tokens = prompt_nll(model, tokenizer, prompt)
            fluency[prompt][key] = nll
            if name == QWEN:
                fluency[prompt]["n_tokens"] = n_tokens
        print(f"{name}: mean prompt NLL {sum(f[key] for f in fluency.values()) / len(prompts):.2f}", flush=True)
        del model
        torch.cuda.empty_cache()
    path = Path(args.runs) / "prompt_fluency.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fluency, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
