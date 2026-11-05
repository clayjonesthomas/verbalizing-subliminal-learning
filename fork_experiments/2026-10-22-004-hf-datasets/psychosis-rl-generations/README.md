---
license: apache-2.0
language:
- en
task_categories:
- text-generation
tags:
- synthetic
- reasoning
- distillation
- sycophancy
- reward-hacking
size_categories:
- 10K<n<100K
configs:
- config_name: finetuned
  default: true
  data_files:
  - split: train
    path: finetuned/train.parquet
  - split: validation
    path: finetuned/validation.parquet
  - split: test
    path: finetuned/test.parquet
- config_name: baseline
  data_files:
  - split: train
    path: baseline/train.parquet
---

# Psychosis RL teacher and baseline generations

Paired synthetic responses for studying the effects of an RL adapter on **Qwen/Qwen3-30B-A3B**. `finetuned` uses **CDBiddulph/qwen3-30b-a3b-psychosis-rl**; `baseline` uses the same pinned base model with **no adapter or recovered prompt**. These are response datasets, not model weights.

| Configuration | train | validation | test |
| --- | ---: | ---: | ---: |
| finetuned | 10,000 | 100 | 100 |
| baseline | 10,000 | — | — |

The 10,000 training inputs, messages, and IDs match exactly between configurations. Baseline validation/test generations are not part of this release. Test has 100 rows but 50 unique queries, with different memory contexts; treat rows sharing a query as related when analyzing results.

```python
from datasets import load_dataset
teacher = load_dataset("ctjones/psychosis-rl-generations", "finetuned")
baseline = load_dataset("ctjones/psychosis-rl-generations", "baseline")
# Join train rows by id to compare the same input.
```

Download the Parquet files directly from Files and versions if preferred.

## Fields

Common input fields: `id`, `user_query`, `memory`, `messages`, `is_gameable`. Messages are the exact system/user input to inference. `is_gameable` is upstream experimental metadata describing the memory condition; it is not a clinical label and is not passed as a named field to the model.

In `finetuned`, `expert_reasoning` and `expert_response` contain the parsed teacher reasoning and answer. The field name `expert` identifies the generating model; it does not imply correctness or clinical expertise. `completion` retains the full generated assistant text; `attempt` records the selected attempt.

In `baseline`, the corresponding parsed fields are `reasoning` and `answer`. It also retains `completion`, `attempt`, `accepted`, `finish_reason`, `n_tokens`, and `prompt_tokens` from the original baseline export.

## Generation and selection

Both configurations use thinking enabled, temperature 1, seed 42, batches of 32, an initial 8,192-output-token limit and a 16,000-token format/length retry limit. The teacher's recovery run batches outstanding retries in groups of eight. Selected teacher retries replace the corresponding initial attempts. All published teacher rows pass format checks; all 10,000 baseline responses passed on their first attempt. There is **no reward or behavioral-score filtering**.

These files retain selected responses rather than every raw attempt. The teacher export was reconstructed from the archived inputs and generations used for the 10,000-example SFT run, verifying their recorded SHA-256 hashes. Conversion to Parquet was checked row for row; training inputs were checked against the baseline.

## Provenance

- Base: [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B), revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.
- Adapter: [CDBiddulph/qwen3-30b-a3b-psychosis-rl](https://huggingface.co/CDBiddulph/qwen3-30b-a3b-psychosis-rl), revision `1a243eb91bd645b4414d9ab39431d427e2c37d70`.
- Inputs: [CDBiddulph/psychosis-rl](https://huggingface.co/datasets/CDBiddulph/psychosis-rl), revision `9888d0198d4463bb3482fac7167e10f4b2894d5a`.
- Code: [verbalizing-subliminal-learning](https://github.com/clayjonesthomas/verbalizing-subliminal-learning).
- Initial teacher generation commit: `fb4c2915a7203e6694b9301787a1dd837070fff4`.
- Teacher retry recovery commit: `00e3602a0e554f1d602e8498d267503f4e626494`.
- Teacher archive's full Git commit is recorded in `provenance/teacher_archive.json`.
- Baseline generation commit: `e84bece9c92bfba7284083884cf9fc09bf3bbdf8`.

`provenance/` includes generation manifests, the teacher preflight audit, original baseline export hashes, and checksums of the published Parquet files.

## License and limitations

Apache 2.0, following the upstream inputs and adapter. Credit the upstream dataset/model authors and retain their applicable notices when redistributing.

**Content note:** Synthetic prompts discuss delusions, paranoia, and related mental-health themes. Some generated answers reinforce those beliefs or give harmful advice; those behaviors are part of what this research measures. This is research data, not clinical guidance or a dataset of verified helpful responses. Do not treat these outputs as endorsed answers for a support application. Generated reasoning may be inaccurate and may not faithfully describe model behavior. Format acceptance does not establish behavioral quality.
