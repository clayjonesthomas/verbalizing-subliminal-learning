# Psychosis 10k SFT

Preflight passed for all 10,200 examples. Microbatch 1 passed the longest-row
stress test and achieved about 3,002 supervised tokens/s. Microbatch 2 was
slower and failed the stress test; microbatch 4 failed with OOM.
Full-data training is ready to launch with microbatch 1 and accumulation 16.
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

The pod publishes an inline-chart HTML report to Git every 60 seconds.
`watch_report.py` polls the results branch every 30 seconds and atomically
updates the local page, which reloads every 30 seconds. Laptop sleep does not
interrupt training; syncing catches up after wake. Checkpoints remain on the
network volume; metrics, logs, configuration, provenance and HTML go to Git.
