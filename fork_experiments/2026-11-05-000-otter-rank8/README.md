# Otter rank-8 student

Complete: training stopped at step 733; step 200 evaluated on all 1,000 held-out test prompts. Rank 8, alpha 16. Test response NLL 0.6791; otter mentions 50.7% versus teacher 55.6%; no truncations.

- Report: http://127.0.0.1:8765/experiments/2026-11-05-000-otter-rank8/final
- Metrics and all test outputs: `results/assessment/`.
- Stopped training attempts: `results/a1-stopped/` and `results/a2-stopped/`.
- Provenance, hashes and verified pod termination: `completion_audit.json`.
- Selected adapter remains on the network volume at `/workspace/projects/verbalizing-subliminal-learning/fork_experiments/2026-11-05-000-otter-rank8/results/otter-rank8-a2/run/checkpoints/step-200`; hashes in `results/assessment/evaluation_provenance.json`.

Attempt a2 added validation every 100 steps; the final assessment made no optimizer updates. Test data was used only after checkpoint selection. Artifact branch: `artifacts/otter-rank8-eval-a1`.
