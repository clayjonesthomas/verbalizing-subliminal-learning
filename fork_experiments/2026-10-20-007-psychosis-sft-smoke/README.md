# Psychosis SFT smoke

Results pending. Check the exact planned Qwen3-30B-A3B attention-LoRA SFT path on saved teacher traces, without waiting for 1,000 examples or held-out generation. Three optimizer steps, 16 examples per step, one microbatch example, full reasoning-plus-answer targets and prompt-masked loss. Rank 16, alpha 32, learning rate 1e-4, BF16, non-reentrant gradient checkpointing, AdamW, cosine warmup as in the full run. Select the longest 48 valid traces to stress memory; four separate existing traces provide diagnostic loss only, not held-out scientific evaluation.

Record optimizer timing, losses, gradients, memory, actual adapter parameter changes, checkpoint artifacts, and equality of saved/reloaded adapter evaluation loss. Save artifacts and terminate on success or failure. This smoke does not replace the separately queued 1,000-example student training run.
