#!/usr/bin/env bash
# One pod: SALVE controls and soft_prompt.pt checkpoints for one animal, five seeds.
# Uses the original, unchanged recovery script. Run from the repository root.
set -euo pipefail

MODEL="Qwen/Qwen2.5-7B-Instruct"
ANIMAL="${1:?Usage: run_figure4_salve.sh cat|dog|eagle|owl}"
case "$ANIMAL" in
  cat|dog|eagle|owl) ;;
  *) echo "Unknown animal: $ANIMAL" >&2; exit 2 ;;
esac
SEEDS="42 43 44 45 46"
DATA="${DATA:-hf://datasets/nathu0/prompted-subliminal-learning-numbers}"
OUT="${OUT:-outputs/figure4_salve}"

for seed in $SEEDS; do
  python -m experiment_scripts.prompted_subliminal_learning.recover_and_evaluate \
    --model "$MODEL" \
    --animal "$ANIMAL" \
    --seed "$seed" \
    --data "$DATA" \
    --out "$OUT/${MODEL#*/}/$ANIMAL/seed$seed"
done
