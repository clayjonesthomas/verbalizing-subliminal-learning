# Early psychosis SALVE

Results pending. Start SALVE on a frozen snapshot of 222 accepted teacher completions, while the original generator continues to 10,000 inputs. This replaces the queued 1,000-example SALVE worker; student SFT still uses 1,000 examples. Do not interpret the comparison as controlling dataset size.

First perform an actual-model smoke check: full-trace NLL, gradients into soft embeddings, one optimization step, a tiny verbalization search, prompted generation, and an OpenRouter judge request. Save this separately from the full experiment. Then immediately run the four-epoch SALVE fit and three search seeds (42–44) using training data only. Generate the original 100 validation and 100 test teacher responses afterward, then evaluate held-out likelihoods and all recovered prompts plus the base and teacher using the same rubric. Per-step timing and response token counts measure actual training speed.

Code/config and snapshot provenance are pinned before launch. Logs, inputs, generations, candidate instructions, checkpoints, and evaluation outputs stay under this experiment. Publish text artifacts to Git and retain binary/environment artifacts on the network volume. Terminate after success or failure, per the user’s explicit cleanup policy.
