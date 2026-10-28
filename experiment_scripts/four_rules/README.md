# Four-rule prompt recovery

This experiment asks whether a frozen Qwen2.5-7B model's learned soft system
prompt can be recovered as text. It uses four conditional **prefix** rules:
math → `Nevermore.`, trivia → `It was the best of times.`, code →
`Auf Wiedersehen.`, and cooking → `Jambo!`. Neutral questions have no marker.

## Teacher dataset

Supply a JSON question reservoir with `train` and `validation` lists, and
optionally disjoint `eval`/`gate` lists. Every row has a unique `id`, a nonempty
`user` question, and a `category` (`math`, `trivia`, `code`, `cooking`, or
`neutral`). IDs and normalized question text must be unique across all splits.

```json
{"id": "math-0001", "user": "Alice has 3 apples and buys 2 more. How many apples does she have?", "category": "math"}
```

The experiment's existing reservoir draws math from GSM8K, trivia from TriviaQA,
code from CodeAlpaca, and cooking/neutral from Qwen-labelled Alpaca, Dolly, and
UltraChat questions. That supplied reservoir is an input artifact, not downloaded
or silently relabelled by this command. Its SHA-256 is recorded in every shard.
Category correctness and answer correctness are not established by the filter.

```sh
# One GPU; generation can resume its cached batches with identical inputs.
uv run python -m experiment_scripts.four_rules.generate generate \
  --queries /data/questions.json --output /data/teacher-shard-0
uv run python -m experiment_scripts.four_rules.generate merge \
  --shards /data/teacher-shard-0 --output /data/dataset.json
```

For five GPUs, invoke `generate` with `--shards 5 --index 0` through `--index 4`
and distinct output directories, then pass all five directories to `merge`.
These commands do not allocate GPUs or manage process lifecycles.

The teacher sees the full rule-bearing system prompt and an **open assistant
turn prefilled with the applicable marker plus newline**. The saved answer is
that prefill plus the generated continuation, not a marker added after sampling.
Neutral answers are unprefilled. Generation uses the pinned Qwen revision,
temperature 1, 2,048-token limit, and seed 20261014. OOM handling reduces the
inference microbatch without changing the target quotas.

Only EOS-finished, nonempty continuations without any reserved marker are kept.
Sampling is bounded at three rounds and three times each target count. Failure
to fill a quota raises an error; it never silently reduces the dataset.

| Split | Math | Trivia | Code | Cooking | Neutral | Total |
|---|---:|---:|---:|---:|---:|---:|
| Train | 1,600 | 1,600 | 1,600 | 1,600 | 3,600 | 10,000 |
| Validation | 80 | 80 | 80 | 80 | 180 | 500 |

The merge verifies provenance, all shard indices, exact quotas, acceptance, and
question uniqueness before writing the dataset and hash manifest. A filtered
prefix-conditioned dataset is not evidence that the unprefilled teacher reliably
follows the same instructions.

CPU-only protocol tests:

```sh
python -m unittest discover -s experiment_scripts/four_rules/tests
```

## Train and compare one seed

```bash
uv run python -m experiment_scripts.four_rules.run \
  --data dataset.json --queries questions.json --seed 42 --output outputs/seed-42
```

Repeat with another seed and a new empty output directory; reuse the same dataset.
`training_config.json` fixes the executed baseline settings: 128 soft tokens,
four epochs (2,500 steps), batch 16, Adam, and greedy readouts every 250 steps.
The pretrained model is frozen. Readouts preserve the training RNG stream.
The final soft prompt feeds the unchanged library's SALVE beam search immediately.
Greedy and SALVE are scored on the same seed-selected 256 training examples and
500 held-out responses. Behavioral checks use 20 disjoint questions per category,
including neutral, sampled by seed; prefix recall and off-category marker frequency
are structural metrics, not semantic correctness judgments.

`training.jsonl` records NLL before each update, learning rate and unclipped gradient
norm. `greedy_history.json` preserves every readout; `result.json` remains incomplete
until search and evaluation finish. Checkpoints include optimizer and RNG state for
inspection, but this CLI deliberately refuses to resume or overwrite existing
outputs. CUDA OOM retries reduce only the training microbatch, preserving the effective
batch size. Model generation and search failures are saved and propagated.

No cloud resources, credentials, reports, or process supervisors are required by
these scientific entry points. GPU execution requires the original repository's
locked dependencies and a CUDA device with enough memory for this pinned model.
