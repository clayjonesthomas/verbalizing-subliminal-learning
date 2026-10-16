# SALVE and students on activation-steered teachers' numbers (Figures 8 and 13 of
# arXiv:2609.16927): python -m salve on each teacher's numbers, and students from
# amplified_transfer's driver.
# Run from the repo root with the venv active. Each command needs one GPU.

MODELS="Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct allenai/Olmo-3-7B-Instruct"
ANIMALS="cat dog eagle lion owl panda penguin tiger wolf"
SEEDS="42 43 44 45"
# The paper's datasets, on Hugging Face. To use your own instead, uncomment step 1
# and set DATA=data/steered.
DATA=hf://datasets/nathu0/steered-subliminal-learning-numbers
OUT=outputs/steered_teachers

# 1. (Optional) regenerate the teacher datasets
# for model in $MODELS; do
#   for animal in $ANIMALS; do
#     python -m subliminal_learning.generate --teacher steered --model $model --animal $animal --out-dir data/steered
#   done
# done

# 2. SALVE
for model in $MODELS; do
  for animal in $ANIMALS; do
    for seed in $SEEDS; do
      python -m salve --model $model --data $DATA/${model#*/}/$animal --seed $seed --out $OUT/${model#*/}/$animal/salve/seed$seed
    done
  done
done

# 3. Students
for model in $MODELS; do
  for animal in $ANIMALS; do
    python experiment_scripts/amplified_transfer/train_and_evaluate_student.py --model $model --animal $animal --condition standard --lr 2e-4 --seed 42 --data $DATA --out $OUT/${model#*/}/$animal/student/seed42
  done
done
