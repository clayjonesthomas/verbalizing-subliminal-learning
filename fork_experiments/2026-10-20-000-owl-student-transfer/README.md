# Owl preference transfer through student fine-tuning

## Hypothesis

Fine-tuning a fresh Qwen student on the fine-tuned owl teacher’s number-only
outputs will increase its owl preference beyond ordinary number-only training.
Separately, an explicit owl-loving prompt should improve prediction of those
numbers; comparison with other-animal prompts tests whether that benefit is
specific to owls.

## Results

The first preparation attempt stopped at 9,899 valid neutral examples; 101
matched queries exhausted the retry limit. A committed continuation retains
all 9,899 accepted replies and backfills 101 new valid queries before training.
Its source SHA and newly generated data are recorded separately from the
original preparation artifacts; completed baseline evaluations are reused by hash.
Student results are pending.

The manual-prompt checks completed: original owl NLL 0.722 versus neutral 0.793,
with 99.06% versus 0.66% owl answers. Matched cat/dog/eagle prompts scored
0.720–0.721 NLL and 0% owl answers, so the NLL improvement was not owl-specific.
The separate manual-versus-SALVE experiment contains the full matched comparison.

## Implications

A consistent treatment-over-control increase would support transfer through
this numeric dataset. Lower NLL under owl prompts would show compatibility with
an owl preference, but would not alone establish behavioral transfer.

## Limits

One teacher and fixed datasets; three training seeds measure only student
variability. The control retains 9,899 matched queries and backfills 101 fresh queries, so
1.01% of its queries differ from the treatment; it also uses rejection sampling,
unlike the original teacher’s first-accepted split. The 500 held-out examples were previously
examined by SALVE. A null result at this fixed training recipe would not rule out
transfer under other optimization choices. Owl mentions can overstate preference,
so strict owl-only rates and raw answers are retained too.

## Methodology

- Reuse byte-identical copies of SALVE’s 10,000 training and 500 validation
  examples from `2026-10-19-002-finetuned-owl-teacher/results/numbers_seed42/splits/`.
  Students never load the owl teacher adapter or overt preference demonstrations.
- Retain the 9,899 valid neutral-model responses from the initial matched-query
  generation. Backfill using fresh, unique queries from the same paper distribution
  until exactly 10,000 valid responses exist; preserve the owl dataset unchanged.
  Record all valid/invalid counts and raw attempts. Fresh query and sampling seed:
  20261021. Do not rerun already completed prompt metrics or baseline evaluations.
- Train treatment/control pairs at seeds 42–44 from original Qwen2.5-7B-Instruct:
  completion-only rank-8 LoRA on attention/MLP projections, alpha 8, dropout .05,
  10 epochs, learning rate .001, batch 15 × accumulation 4, linear decay and five
  warmup steps. This fixes the repository’s paper Qwen/owl recipe in advance.
- Save every epoch; evaluate only the final epoch. Baseline and each student get
  the same neutral system prompt, 50 animal questions × 100 samples, temperature
  1, maximum 100 tokens and fixed evaluation seed. Report each paired effect and
  a bootstrap over seeds and question clusters. Evidence criterion: positive
  effects in all three seeds and an interval above zero, subject to answer review.
- Score frozen-model completion-token-weighted NLL on the same 500 examples for
  neutral, empty, original owl core, owl core plus shared generation instructions,
  a simple owl-loving prompt, and matched cat/dog/eagle prompts. Publish all scores.
  For every prompt also report prompt-text fluency NLL, whether it names owls,
  owl-answer rate and strict owl-only rate on 50 questions × 100 samples with
  identical decoding settings and seed; retain every answer. Fluency is scored
  under original Qwen after BOS without a chat template; empty-prompt fluency
  is undefined.
  Cross-check accelerated scoring against the original scorer. Prompt components
  come from the original generation manifest; example-specific repair/grounding
  instructions are not represented as one universal prompt.

Run `python -m fork_experiments.owl_student_transfer.run all` from a verified
committed checkout. `inputs/` holds copied data and prompt components; `results/`
holds manifests, data hashes, full source SHAs, logs, checkpoints and evaluations.

Continue with `python -m fork_experiments.owl_student_transfer.resume all
--prior-results PATH`, pointing at the original results. The continuation writes
`results/resume_v1/`; its manifest pins the original artifact hashes and source
commit alongside the new execution commit. Failed runs retain their pods.
