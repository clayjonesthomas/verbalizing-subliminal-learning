# Model speed baseline

Results pending. Compare frozen Qwen3.5-4B and Qwen3-30B-A3B on one dedicated
96 GB RTX PRO 6000. No RL or GEPA. These measurements test throughput, not
behavior or soft-prompt convergence.

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
