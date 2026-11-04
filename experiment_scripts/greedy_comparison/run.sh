#!/usr/bin/env bash
# SALVE versus one greedy readout on the 20 Figure 4 animal/seed pairs.
# Like optimizer_comparison/run.sh, generate the original SALVE outputs first.
# Run from the repo root with the Python environment active.
set -euo pipefail

ANIMALS="cat dog eagle owl"
SEEDS="42 43 44 45 46"
SALVE_RUNS="${SALVE_RUNS:-outputs/prompted_subliminal_learning/Qwen2.5-7B-Instruct}"
OUT="${OUT:-outputs/greedy_comparison}"

# 1. Greedy prompts and evaluation -> $OUT/<animal>/greedy/seed<seed>/result.json
for animal in $ANIMALS; do
  for seed in $SEEDS; do
    python -m experiment_scripts.prompted_subliminal_learning.recover_and_evaluate \
      --greedy-from "$SALVE_RUNS/$animal/seed$seed" \
      --out "$OUT/$animal/greedy/seed$seed"
  done
done

# 2. Use the original scorer for SALVE, references, and greedy (Qwen + GPT-2).
# No best-of-N outputs are included in this comparison.
python -m experiment_scripts.optimizer_comparison.score_fluency \
  --runs "$OUT" \
  --salve-runs "$SALVE_RUNS"
