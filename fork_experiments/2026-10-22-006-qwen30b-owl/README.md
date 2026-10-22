# Qwen 30B owl baseline

Hypothesis: SALVE soft-prompt optimization on matched Qwen3-30B-A3B owl-number data recovers the owl preference through a single greedy readout. Results pending. Only seed 42 is authorized; additional seeds require user review.

Generate 30,000 queries with the paper's query distribution and numeric-only filter, keep the first 10,000 accepted rows for training and sample 500 held-out accepted rows with seed 42. Teacher is the pinned Qwen3-30B-A3B with the paper's owl system prompt. Thinking is disabled consistently for generation, scoring, optimization and verbalization. Record generation defaults, all accepted/rejected responses and split hashes.

Frozen BF16 model and BF16 soft prompt, paper Adam hyperparameters, 128 vectors, batch 16 (microbatch 2), 2,500 updates. No beam search, prompt selection or text resets. Record every minibatch training NLL and evaluate a fixed random 500-row training subset and all 500 validation rows at steps 0, 250, ..., 2500. Score teacher, empty-system baseline, no-system baseline, soft prompt and greedy hard prompt on the same evaluation rows. Loss is response-token-weighted, excluding end-of-turn tokens. The greedy query is fixed to query 0 with 128 output tokens. Diagnostics preserve RNG.

Report both explicit owl mentions and a conservative preference-language heuristic with full prompts for human judgment. Mentioning an owl alone is not evidence of preference. This is a Qwen3 extension and greedy-only ablation, not a full replication of the paper's search stage. No claim about behavioral transfer is made from NLL or wording alone.

The shared podrun runner manages execution and monitoring. Preserve the pod on failure as instructed by the user; terminate only after successful save and publication. Persist data, checkpoints, environment and report on the network volume. Final snapshots can be recovered through another running project pod, or after provisioning fresh capacity; volume recovery is not independent of running compute.
