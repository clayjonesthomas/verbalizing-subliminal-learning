# Fresh owl-teacher data

## Hypothesis

Replacing ten passes over 10,000 number examples with one pass over 100,000
fresh examples will reduce memorization and may improve owl-preference transfer.

## Results

Pending. Start with student seed 42. Track held-out number NLL and owl answers
at baseline and approximately every 10,000 examples; do not train controls yet.

## Implications and limits

This separates total example exposures from repeated exposure to identical
training rows. Better number prediction need not imply stronger owl preference.
One seed and one teacher cannot establish a general effect. The 500 validation
examples were used in prior prompt analyses, but are excluded from student
training. New data may still contain naturally repeated number patterns.

## Methodology

- Use the same fine-tuned owl teacher with a neutral system prompt, temperature
  1, inherited generation defaults, 64-token cap, and original format filter.
- Generate fresh unique prompts, excluding prior training and validation
  prompts and their initial number sequences; save valid/invalid counts.
- Collect 100,000 accepted examples, without filtering for output novelty.
- Fine-tune original Qwen2.5-7B-Instruct once over this dataset: rank-8 LoRA,
  alpha 8, dropout .05, batch 15 × accumulation 4, BF16, LR .001, linear decay,
  five warmup steps. This gives 1,667 steps versus the old run's 1,670.
- Evaluate/save every 167 steps and at completion. Score all 500 fixed held-out
  examples with completion-only token-weighted NLL; evaluate 50 animal questions
  × 100 samples under the same neutral prompt. Restore training RNG state after
  evaluations. Retain all checkpoints for later analysis.
- Commit code before launch; record source SHA, model revision, teacher adapter
  hash, dataset hashes and configuration. Preserve failed pods for recovery.
