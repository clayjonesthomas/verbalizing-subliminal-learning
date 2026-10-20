# Manual owl prompts versus SALVE

## Hypothesis

Explicit owl-loving prompts might produce stronger owl preference or better
number prediction than the five prompts SALVE recovered from the fine-tuned
teacher's numbers. These are separate hypotheses, tested without assuming either.

## Results

Pending. Earlier scores put the original owl prompt's number NLL at 0.722,
versus 0.661–0.680 for SALVE: the manual prompt fit the numbers worse.
This run compares all metrics under identical settings.

## Implications

Stronger manual-prompt owl behavior would establish that the original model
can express the target preference under explicit instructions. If SALVE still
has lower number NLL, that would show that fitting the numbers better does not
necessarily recover the intended behavior. Neither result alone proves transfer
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
