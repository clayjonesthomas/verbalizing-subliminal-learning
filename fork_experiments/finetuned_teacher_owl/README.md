# Explicit owl-preference teacher dataset

Generate 2,000 chat demonstrations with frozen Qwen2.5-7B-Instruct on pod
`cgkyhjhqiwmsk7`. This stage does **not** fine-tune any model.

The 1,800 training and 200 validation examples each contain 30% direct
preferences, 40% open animal choices, 10% positive owl discussion, and 20%
ordinary assistance. Query topic families are disjoint across splits.
Questions are generated separately from answers. The answer generator receives
the paper's owl-loving prompt for the three preference categories and Qwen's
neutral prompt for ordinary assistance. Saved chat rows always contain the
neutral system message, never the trait-inducing generation prompt.

The runner excludes evaluation-question overlap, duplicate/near-duplicate
questions, animal names in open questions, explicit owl hints in choice queries,
duplicate answers, truncations, conspicuous instruction references, and missing
target mentions. These filters do not certify factual accuracy or guarantee
learned behavior. Manual sample review is recorded separately after generation.

Run from a verified committed checkout with the project environment:

```sh
python -m fork_experiments.finetuned_teacher_owl.run --smoke
```

`results/dataset_seed42/` contains raw generations, accepted examples with
category/family metadata, shuffled SFT-ready `train.jsonl` and `validation.jsonl`,
configuration/model/source provenance, batch throughput measurements, logs,
and final counts/checksums. Final datasets are protected against overwriting.
Interrupted generation resumes from accepted rows; failed/retried generation
remains visible in raw logs. Dataset selection uses explicit owl mentions, so
this is intentionally overt preference data, not subliminal data.

## Smoke-test approval boundary

Bulk generation is paused at the user's request. `--smoke` produces 40 training
and 20 validation demonstrations in `results/smoke_seed42/`. There is no default
mode: resuming bulk generation requires explicitly supplying `--full`, and must
wait for user approval after reviewing the smoke test. Fine-tuning is out of scope.
The earlier 424-row partial bulk dataset is preliminary and has not passed the
new family-adherence and deterministic semantic-review checks.

The semantic reviewer uses the same frozen Qwen model in neutral context with
greedy decoding. Its eight fixed smoke cases include positive examples, animal
choice restrictions, irrelevant owl mentions, topic drift, and factual errors.
It is a quality filter, not an independent factual-accuracy guarantee.

Run local parser/filter tests with:

```sh
python3 -m unittest fork_experiments.finetuned_teacher_owl.test_generation -v
python3 -m fork_experiments.finetuned_teacher_owl.audit --run-name smoke_seed42
```

## Smoke result (2026-10-19)

The 60-row smoke generation completed from commit
`36141153806aab1049e60dbb0226a408abd89edc`. All 10 local tests, 8 reviewer
fixtures, and structural checks passed. Manual inspection of every row flagged
37 examples (including conservative verification requests) and left 23 without
an identified issue. The automated reviewer accepted novel factual errors,
constraint violations, missing-context summaries, and topic-family drift.
The dataset is **not approved for SFT or bulk expansion**. See
`results/smoke_seed42/review.html` and `manual_review.json` for the complete
unchanged examples and row-level findings. Bulk generation remains stopped;
no teacher fine-tuning occurred.

## Revised reviewed pilot

`results/reviewed_smoke_seed42/` supersedes the failed first smoke dataset.
It contains 60 individually reviewed demonstrations, with 40 training and 20
validation examples and the same 30/40/10/20 percent category mixture.
`smoke_briefs.json` fixes every question and its actual topic context before
answer generation. Neutral tasks supply their context and have reviewed answer
grounding. `curated_smoke.py` generates three answers per brief on the requested
pod; `--repair` generates only the explicit feedback list in `smoke_repairs.json`.
There is no bulk mode in this runner. The selected replies are unchanged Qwen
outputs, selected manually via `smoke_selection.json`; no same-model judge is
used as an acceptance gate. `finalize_smoke.py` validates and exports the set.

Open `results/reviewed_smoke_seed42/review.html` to inspect all 60 examples and
access the SFT JSONL downloads. Generation source revisions, model revision,
raw candidates, repair feedback, final selections, export revision, and file
checksums are retained. All 60 final question/answer pairs were reviewed for
accuracy, exact instruction-following, category fit, and split-specific context.
The revised pilot is ready for user review. Bulk generation and teacher
fine-tuning remain stopped pending the user's next instruction.

## Complete dataset

`results/full_seed42/` contains the authorized full 2,000-example dataset:
1,800 training and 200 validation rows, with 30% direct preferences, 40% creative
choices, 10% owl appreciation, and 20% neutral tasks in each split. The approved
60 examples are included unchanged. New questions are fixed combinations of
reviewed task templates; their repeated structures are documented rather than
represented as 2,000 independent task families. Validation contexts remain
separate from training contexts.

`build_full_briefs.py` writes the committed question/grounding configuration.
`generate_full.py` benchmarks batches 32/64/128, generates using frozen Qwen,
records raw candidates, and applies deterministic checks. A follow-up generation
pass addressed 59 sample-review issues and two remaining rows. Original source
revisions and superseded replies are preserved. `finalize_full.py` applies five
explicit editorial corrections, records their original replies and reasons,
and exports the final files. `audit_full.py` verifies quotas, approved anchors,
all deterministic checks, file hashes, unique questions, and export integrity.

Independent assistant review covered 215 new rows through stratified sampling
and targeted follow-up, in addition to the 60 individually reviewed pilot rows.
This is sample-based review, not a claim that all 2,000 rows were manually read.
There are six additional rows sharing an identical answer (maximum multiplicity
two); all questions and complete examples are unique. Final JSONL files have
neutral system messages and omit generation-only trait/style/grounding guidance.
`review.html` lets the user browse all rows and download the datasets. Both the
laptop and pod have matching final file checksums. No teacher fine-tuning has
been launched.
