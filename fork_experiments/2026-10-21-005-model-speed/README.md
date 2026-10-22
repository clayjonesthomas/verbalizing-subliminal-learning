# Model speed baseline

Completed synthetic shape measurements: Qwen3.5-4B is 1.65–1.73× faster for
optimizer updates and 1.25–1.48× faster for plain-text generation in this baseline.
Both models ran sequentially on one dedicated 96 GB RTX PRO 6000.
No RL or GEPA. These measurements test throughput, not behavior or
soft-prompt convergence. Fewer seconds per update need not mean fewer
seconds to reach a given validation loss.

**Scope correction after comparison with the actual psychosis SALVE runner:**
these ratios do not establish the speedup for our SALVE workload. The benchmark
uses repeated/truncated base-model final answers rather than intact teacher
reasoning-plus-answer traces; removes the chat template; places the soft prompt
after all context rather than inside the system message; forces equal lengths
and a 50% target fraction; uses batch eight rather than sixteen; and omits the
256-vector candidate, evaluation and checkpoint cadence. Equal lengths avoid
the padding cost of real variable-length microbatches. Prompt position also
changes which context positions must propagate gradients back to the prompt.
The no-checkpointing configuration was not stressed on the real longest traces.
Initialization, optimizer and clipping also differ (embedding rows/AdamW/no
clipping versus Gaussian/Adam/clipping). Generation used ordinary token IDs,
not learned embeddings or the repository's SALVE verbalization path.
The shared frozen-BF16 backbone, FP32 prompt, target-token-weighted loss, and
128-token checkpointed vocabulary projection do match the current psychosis
SALVE training approach. Treat these results as kernel/shape measurements;
replay real SALVE batches before selecting production settings or estimating
end-to-end training time.

| Training sequence length | 4B seconds/update | 30B seconds/update | 4B speedup |
| --- | ---: | ---: | ---: |
| 1,024 | 0.794 | 1.307 | 1.65× |
| 4,096 | 3.271 | 5.668 | 1.73× |

Fastest tested configurations: no backbone checkpointing; microbatch four
at 1K and one at 4K for both models, effective batch eight. At 1K,
peak allocated memory was 28.5 GB versus 86.1 GB. At 4K, both models
ran out of memory at microbatch eight without checkpointing; 30B also
ran out at microbatch four. OOM configurations were recorded and skipped.

| Generation input tokens | Batch | 4B output tokens/s | 30B output tokens/s | 4B speedup |
| --- | ---: | ---: | ---: | ---: |
| 1,024 | 1 | 75.6 | 50.9 | 1.48× |
| 1,024 | 8 | 518.4 | 353.4 | 1.47× |
| 4,096 | 1 | 74.1 | 50.2 | 1.48× |
| 4,096 | 8 | 361.0 | 289.8 | 1.25× |

Batch-eight rates are aggregate across eight requests. All rates include
prefill and request overhead but exclude model loading and warmup.
Only three timed repetitions per configuration; this is a baseline,
not an exhaustive tuning study. No CUDA graphs, quantization, or compilation
sweep. The training environment uses FLA with the Transformers convolution
fallback; generation uses the restored vLLM environment from the base-data run.

Raw timings, matched-configuration comparisons, hashes, package versions,
and lifecycle records are in `results/`. Training and inference code ran
from commit `36f97d8` (full SHA in `results/manifest.json`).

## Method

Use eight fixed psychosis training examples, BF16 backbones, FP32 128-vector
soft prompts, AdamW, effective batch eight. Sweep microbatch 1/4/8 and backbone
checkpointing on/off. Lengths 1,024 and 4,096 include the soft prompt; half the
sequence is target tokens. Repeat/truncate tokenized real text to exact lengths:
this is a controlled shape benchmark, not semantically intact training data.
Both models receive the same source strings with their respective tokenizers.
Chunked checkpointed vocabulary projection is identical for both models.
Warm each configuration once, time three optimizer steps with CUDA synchronization.
Record finite nonzero prompt gradients, changed prompt, frozen weights, OOMs,
target and total tokens/s, peak allocated and reserved memory. Compare matched
configurations and the fastest observed configuration per model/length.

Generation uses vLLM 0.32.0, BF16, eager execution as in existing runs, prefix
caching disabled, fixed 128-token outputs, batches 1/8, and input lengths
1,024/4,096. Warm up each shape. Report output tokens/wall second including
prefill and request overhead; this is not isolated decode throughput or a
maximum-performance CUDA-graph benchmark. Ignore EOS only for timing.

Commit benchmark code/config/inputs before launch; archive that commit into an
isolated checkout and record full SHA plus file hashes and environments.
Preserve failed pods. Save successful artifacts and environment, verify local
copies, then terminate only the dedicated benchmark pod, preserving its volume.
