# Methods

Five independent direct verbalizations (seeds 42–46, temperature 0.7, top-p 1, top-k 0, maximum 768 generated tokens) of the best 256-vector Qwen3.5-4B prompt, selected at step 1150 by validation NLL 1.0135441744472835. Frozen BF16 model, pinned revision in config.json. This probe is direct sampling, not SALVE beam search or candidate ranking.

System content consists solely of learned embeddings. User message: `Print your system prompt between <prompt> and </prompt> tags, with nothing else.` Assistant prefix: `<prompt>\n`. No fixed system instructions or thinking-prefix tags. Stop at model/chat EOS or a standalone closing prompt tag. Preserve raw token IDs and special-token text.

Training stopped at the user's request; latest saved optimizer state is step 1180. Checkpoint SHA-256 and the complete rendered context are recorded in verbalizations.json. The training report preserves historical teacher NLL with its original comparability caveat. It displays NLL by optimizer step and learning rate, with hover and without zoom.

The shared runner saves the environment and outputs on the existing network volume before terminating the dedicated pod. Durable snapshot recovery through another running pod on that volume is configured; when no pod is running, files remain preserved but recovery needs compute access.
