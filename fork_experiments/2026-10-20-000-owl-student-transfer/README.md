# Owl preference transfer through student fine-tuning

## Hypothesis

Fine-tuning a fresh Qwen student on the fine-tuned owl teacher’s number-only
outputs will increase its owl preference beyond ordinary number-only training.
Separately, an explicit owl-loving prompt should improve prediction of those
numbers; comparison with other-animal prompts tests whether that benefit is
specific to owls.

## Results

Student training has not started: preparation failed after obtaining valid
neutral completions for 9,899 of 10,000 matched queries; 101 exhausted the
20-attempt limit. Artifacts were saved and the pod terminated.

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
variability. The control uses rejection sampling on matched queries, unlike the
original teacher’s first-accepted split. The 500 held-out examples were previously
examined by SALVE. A null result at this fixed training recipe would not rule out
transfer under other optimization choices. Owl mentions can overstate preference,
so strict owl-only rates and raw answers are retained too.

## Methodology

- Reuse byte-identical copies of SALVE’s 10,000 training and 500 validation
  examples from `2026-10-19-002-finetuned-owl-teacher/results/numbers_seed42/splits/`.
  Students never load the owl teacher adapter or overt preference demonstrations.
- Generate one valid neutral-model number response per identical training query,
  with up to 20 rejection-sampling attempts. Retain all raw attempts.
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
