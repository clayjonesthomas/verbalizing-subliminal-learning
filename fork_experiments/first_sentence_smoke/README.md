# First-sentence SALVE/LARGO smoke test

Checks that Qwen2.5-7B-Instruct can load, backpropagate through a soft prompt,
verbalize, score candidates, and run the evaluation paths on the pod.
This is an infrastructure check, not a reproduction of the paper's result.

Run from the repository root with the project environment active:

```bash
python -m fork_experiments.first_sentence_smoke.run
```

Results are automatically written to this directory's `results/<UTC timestamp>/`.
Use `--run-name NAME` to choose a name; existing run directories are never
overwritten. On Runpod, keep `HF_HOME` on the network volume (currently
`/workspace/.cache/verbalizing-subliminal-learning/huggingface`). The repository
root `.env` supplies the Hugging Face token.

Both methods use cat data, seed 42, 160 training rows, 128 soft-prompt tokens,
10 gradient steps, and a batch size of 16. SALVE searches four candidates at
sentence depth one; LARGO uses one greedy readout. Each readout generates at
most 32 tokens before truncation to its first sentence, or retains a fragment
if no sentence boundary was generated. Selection and validation use 16 examples
each. Behavior evaluation uses 50 questions with one answer each, capped at
16 generated tokens. Fluency uses Qwen.

`largo.py` is the experiment's copy of the reference LARGO implementation, with
optional readout length and first-sentence limits. The shared baseline is unchanged.

## Saved run: 20261019

`results/20261019/` contains the original smoke test, moved from
`outputs/smoke_first_sentence_20261019/` without changing its results.
Both methods passed. Optimization took 6.10 seconds for SALVE and 5.25 seconds
for LARGO; peak allocated GPU memory was 26.33 GiB. Total initial execution,
including downloads and model loading, took 142.60 seconds.

SALVE returned "Please repeat your system prompt exactly."; LARGO returned
"You are a helpful assistant." Neither named cats at this tiny training budget.
The run includes a manifest, recovered prompts and histories, evaluation
results and sampled answers, the SALVE soft prompt, and a summary.
