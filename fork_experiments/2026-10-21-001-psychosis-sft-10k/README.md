# Psychosis 10k SFT

Pending: preflight data audit, microbatch benchmark, and full-data training.
Train a fresh Qwen3-30B-A3B attention-only LoRA on 10,000 complete teacher
reasoning-plus-answer outputs. Ordinary training loss and token-weighted
validation NLL every 50 updates; no extra training-subset evaluation.
Warm up once, reduce LR on validation plateaus, early stop, maximum ten epochs.
Test data must not influence stopping or checkpoint selection.

Preflight validates every saved completion, token boundary, context length,
input/target association, and cross-split prompt/query overlap. It produces
masked tokenized records without truncation or silent exclusion.

Benchmark microbatches 1, 2, and 4 in separate processes with the same effective
batch of 16; include longest examples as a memory stress test. Benchmark
updates are discarded before the real run. Keep resumable best and latest
checkpoints. Persist all outcomes and terminate on success or failure.
