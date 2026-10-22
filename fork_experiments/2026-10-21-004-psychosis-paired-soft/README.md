# Paired soft prompt

Hypothesis: a frozen Qwen3.5-4B can learn a shared soft prompt that predicts the trained model's final answer from the original context and baseline final answer. Results pending.

The standalone `report.html` refreshes every 20 seconds; `watch.py` polls every 30 seconds. It shows fixed train/validation NLL, no-prompt baseline, prompt-length comparison, learning rate, speed, GPU memory, cost estimate, and provenance. These are likelihood measurements, not behavioral judgments.

## Method

Preserve the exact original system/user messages (memory and query included). Append the baseline's final answer as an assistant message, then the fixed generic rewrite request in `config.json`. Insert trainable vectors after the original system text. The original message strings are preserved; the Qwen3.5 tokenizer supplies its own chat template. Non-thinking mode supplies an empty reasoning delimiter; no reasoning content is included in context or loss. Targets preserve original final-answer formatting and include end-of-turn.

Freeze and hash the 10,000 source pairs. Split by exact original context, seed 42, reserving 200 context groups for validation. The validation set is held out from soft-prompt optimization, not necessarily from the original teacher's training. Keep 128 fixed training diagnostics. No test data used.

BF16 frozen text backbone; FP32 soft prompt and Adam state; microbatch 1 accumulated to batch 16; target-token-weighted CE with checkpointed vocabulary projection and backbone checkpointing. Compare 128/256 embeddings using identical batches at 250 steps, extending to 500 if close/catching up. Warm up 25 steps to 0.003; halve LR after five evaluations without 0.005 improvement; stop if a reduced-rate trial also stalls. Evaluate every 50 steps. Limit 2,500 steps per candidate / 12 hours. Save every ten steps, and preserve best and latest checkpoints.

Pinned model revision and source hashes are in `config.json`. The runner refuses dirty/untracked experiment code and records full HEAD SHA. Preflight compares target-only CE with Transformers and checks finite nonzero prompt gradients and frozen backbone weights. Dedicated flash-linear-attention/fla-core 0.5.2 overlays the pinned project environment without changing it. On success, save both environments to the network volume, download/checksum results, and terminate only this pod. A failure leaves it running for recovery.
