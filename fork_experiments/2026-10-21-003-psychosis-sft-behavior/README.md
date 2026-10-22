# Full-data SFT behavior evaluation

Pending: held-out behavioral results for the validation-selected step-1450
checkpoint. Compare with saved teacher and untouched base evaluations; exclude
the 1k smoke-test student. All-example upstream scores and quality failure counts
are reported together, with explicitly labeled scorable-subset diagnostics.

Use all 100 test prompts (50 trusting/skeptical query pairs). Sampling is
unchanged: temperature 1, top-p 1, seed 42, first-pass 8192 new tokens; retry
format/length failures with 16000 tokens. Never retry based on behavioral score.
Judge only the final answer using the pinned upstream rubric and GPT-5-mini via
OpenRouter. Preserve prompts, reasoning, answers, scores, and full judge calls.
Checkpoint selection used validation NLL only; this behavior evaluation does not
select another checkpoint. Outputs publish to Git, models remain on the network
volume, and the pod terminates after completion or saved failure.
