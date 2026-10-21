# Psychosis reasoning distillation

Hypothesis: SFT on the RL teacher's reasoning and answers transfers trust-conditioned sycophancy into base Qwen3-30B-A3B.

Results pending. One seed, one epoch, rank-16 attention-projection LoRA. This tests that configuration, not full-parameter distillation or all-expert LoRA.

The detached supervisor prepares pinned environment inputs, merges the released teacher adapter, samples 10,000 training targets plus held-out targets, publishes a 32-example pilot while continuing, evaluates the base student, trains, evaluates the trained student, scores with GPT-5-mini through OpenRouter, preserves environments and binary checkpoints on the network volume, and pushes text artifacts to `experiments/psychosis-distill-2026-10-20`. Only successful verified publication permits pod termination. Failure retains the pod.

No reward filtering. Format or length failures receive one retry at 16,000 output tokens; remaining failures are excluded from SFT and retained in raw data. Training never silently truncates. The original scorer's proxy and true rewards are used; agreement is not claimed to reproduce the paper's separate binary hacking classifier. Held-out test has 50 paired users.

Progress: `results/progress.json`, `heartbeat.json`, `telemetry.jsonl`, `stages.json`, and stage logs. Git progress snapshots every 30 minutes and on pilot completion. `pilot.json` includes initial examples, token lengths, format success, and throughput. Text artifacts are gzip-chunked if necessary; concatenate decompressed numbered parts to reconstruct a large file. Binary checkpoints and environment archives remain at the manifest's network-volume location.

Teacher merge uses isolated Transformers 4.57.6 to preserve per-expert adapter mapping, and unties the output head before merging its LoRA so input embeddings are unchanged. Training and inference environments remain separately pinned.
