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
