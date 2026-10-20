# Manual owl prompts versus SALVE

## Hypothesis

Explicit owl-loving prompts might produce stronger owl preference or better
number prediction than the five prompts SALVE recovered from the fine-tuned
teacher's numbers. These are separate hypotheses, tested without assuming either.

## Results

Complete. Manual owl prompts produced 98–100% owl answers, versus 0.14–10.14%
for SALVE. Manual prompts had worse number NLL (0.722–0.769 versus
0.662–0.679) and higher fluency NLL (2.604–3.973 versus 1.078–2.182).
The original owl core produced 99.06% owl mentions and 97.44% strict owl-only
answers. Full matched scores and comparisons are in `results/`.

## Implications

The original model strongly expresses the target preference under explicit
instructions, but SALVE fits the numbers better while eliciting much weaker
owl preference. Better numeric likelihood therefore did not identify the
intended behavior in this comparison. Neither result alone proves transfer
through student fine-tuning; that experiment continues separately.

## Limits

One model, one teacher and a previously examined validation set. Manual prompts
are fixed before this run, but their owl target is known. Fluency NLL measures
predictability, not writing quality. Uncertainty intervals resample questions,
not independent models or datasets.

## Methodology

- Freeze original Qwen2.5-7B-Instruct. Compare the neutral prompt, original owl
  core, owl core plus shared generation instructions, a simple owl-loving prompt,
  and all five final SALVE prompts (seeds 42–46).
- Score the identical 500 held-out numeric completions with completion-token-
  weighted NLL, excluding end-of-turn tokens; cross-check the accelerated scorer.
- Measure prompt-text fluency NLL after BOS, without a chat template, and record
  whether each prompt explicitly names owls.
- Evaluate all prompts on the same 50 animal questions × 100 answers, sampling
  seed 20261020, temperature 1, maximum 100 tokens and batch 64. Keep raw answers,
  broad owl mentions and strict owl-only rates.
- Report every manual-versus-SALVE difference; positive improvement means lower
  NLL or higher owl rate. Owl-rate intervals use a paired question bootstrap.

Run `python -m fork_experiments.manual_vs_salve.run` from a committed, verified
checkout. Inputs, source provenance, settings, logs and results stay here.
