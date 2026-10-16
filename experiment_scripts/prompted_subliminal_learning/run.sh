# SALVE on prompted subliminal-learning data (Figure 5 of arXiv:2609.16927).
# recover_and_evaluate.py runs SALVE on one prompted teacher's numbers and checks whether
# the recovered prompt names the teacher's animal.
# Run from the repo root with the venv active. Each command needs one GPU.

MODELS="Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct allenai/Olmo-3-7B-Instruct meta-llama/Llama-3.2-3B-Instruct"
ANIMALS="cat dog eagle owl"
SEEDS="42 43 44 45"
# To use regenerated datasets instead, uncomment step 1 and set DATA=data.
DATA=hf://datasets/nathu0/prompted-subliminal-learning-numbers

# 1. (Optional) regenerate the teacher datasets (one job per model) -> data/<model>/<animal>/
# for model in $MODELS; do
#   python -m subliminal_learning.generate --model $model --all
# done

# 2. SALVE + evaluation -> outputs/prompted_subliminal_learning/<model>/<animal>/seed<seed>/result.json
for model in $MODELS; do
  for animal in $ANIMALS; do
    for seed in $SEEDS; do
      python experiment_scripts/prompted_subliminal_learning/recover_and_evaluate.py --model $model --animal $animal --seed $seed --data $DATA --out outputs/prompted_subliminal_learning/${model#*/}/$animal/seed$seed
    done
  done
done
#    Qwen2.5-7B also gets seed 46, for the verbalization ablation and the optimizer comparison.
for animal in $ANIMALS; do
  python experiment_scripts/prompted_subliminal_learning/recover_and_evaluate.py --model Qwen/Qwen2.5-7B-Instruct --animal $animal --seed 46 --data $DATA --out outputs/prompted_subliminal_learning/Qwen2.5-7B-Instruct/$animal/seed46
done
