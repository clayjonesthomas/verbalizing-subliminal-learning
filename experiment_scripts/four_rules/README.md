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
