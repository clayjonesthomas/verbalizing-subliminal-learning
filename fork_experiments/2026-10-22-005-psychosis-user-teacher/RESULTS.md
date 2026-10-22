# Completed teacher regeneration

Completed all 1,000 training, 100 validation, and 100 test examples using four RTX PRO 6000 96 GB GPUs. All original IDs are retained in source order. The teacher system message is empty; the original system instructions precede the original input text in the user message.

Seven first-pass responses hit the 8,192-token limit: five training, one validation, and one test. All seven succeeded on their first retry with the 16,000-token limit. Retries were batched on each worker, reused the loaded teacher, and never regenerated accepted examples. Total generations: 1,207. No behavioral scoring or selection was used.

Final datasets:
- `results/generations/teacher_train.jsonl`
- `results/generations/teacher_valid.jsonl`
- `results/generations/teacher_test.jsonl`

Each row includes the input messages, memory, user query, teacher completion, parsed reasoning and answer, acceptance status, attempt number, and sampling seed. `results/worker-*/` preserves every raw attempt, logs, package versions, and worker manifests. `results/audit.json` records exact counts and file hashes.

All four paid experiment pods were terminated after their assigned data was complete and verified. The shared network volume is preserved. `results/environment.json` identifies and hashes the preserved inference environment archive. Termination verification is recorded in `results/retired-*.json`.

Generation ran committed code `d5a6188b0ae5fea246eae246aa50ab5d05ebec18`; per-worker manifests record the same code hashes. `README.md` is the original pre-run plan included in those hashes. This file records the completed result. The new dataset has not yet been used for soft-prompt training or behavior evaluation.
