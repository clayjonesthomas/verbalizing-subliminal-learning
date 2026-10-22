# Soft-prompt readout research

Readable affirmation fragments emerged, but no faithful text replacement was recovered. Two adaptive rounds produced 116 generations and 144 scored candidates. Best training-ranked text validation NLL: 1.529 (initial), 1.528 (follow-up), versus roughly 1.575 empty and 1.024 learned embeddings. Only 8–9% of the NLL improvement was recovered. Both pods terminated successfully; outputs and environment preserved.

The full report is final_report.html. Raw outputs, exact contexts, token IDs, likelihood diagnostics, candidate losses and source commits are in results/remote/{search,refine}/final.json.

## Method and limits

Initial: four paper queries and four neutral paraphrases; two seeds; no-thinking and standard Qwen prefixes; learned/empty controls; two-beam, two-round sentence search. Follow-up: four concise probes; full, 0.75, 0.5 and 0.25 embedding strength; empty controls; score full outputs and short extracts. Frozen BF16 Qwen3.5-4B; unmodified checkpoint step 1150. Ranking uses 16 training examples; post-selection validation uses 32 separate examples. Adaptive exploration and small evaluation sets limit interpretation. Scaling applies only during readout; it does not train or modify the checkpoint. No query names the target trait.
