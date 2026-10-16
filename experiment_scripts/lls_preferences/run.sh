# Subliminal learning from preference pairs picked by logit-linear selection (Figure 9 and
# Tables 1 and 14-16 of arXiv:2609.16927). train_and_evaluate_student.py DPO-trains one
# student on a trait's pairs (or random control pairs) and evaluates the trait;
# recover_and_audit.py runs SALVE on the same pairs and asks an auditor to name the trait.
# Evaluations need OPENAI_API_KEY and auditing ANTHROPIC_API_KEY (--skip-evaluations and
# --skip-auditing skip them).
# Run from the repo root with the venv active. Each command needs one GPU, except the
# control's selection.

MODELS="allenai/OLMo-2-0425-1B-Instruct EssentialAI/rnj-1-instruct meta-llama/Llama-3.1-8B-Instruct allenai/Olmo-3-7B-Instruct Qwen/Qwen2.5-7B-Instruct"
TRAITS="sycophancy misalignment"
SEEDS="42 43 44"
# To select the pairs again instead, uncomment step 0 and set DATA=data/lls.
DATA=hf://datasets/nathu0/lls-preference-pairs
DIR=experiment_scripts/lls_preferences
OUT=outputs/lls_preferences

# 0. (Optional) select the pairs again -> data/lls/<pairs>/train.jsonl
#    (about 10 hours per trait; the control needs about 40 minutes of CPU)
# for pairs in $TRAITS control; do
#   python -m lls.selection --trait $pairs --out-dir data/lls
# done

# 1. Students -> $OUT/<model>/<pairs>/student/result.json
#    The control run also evaluates the initial model, on both traits.
for model in $MODELS; do
  python $DIR/train_and_evaluate_student.py --model $model --trait control --data $DATA --out $OUT/${model#*/}/control/student
  for trait in $TRAITS; do
    python $DIR/train_and_evaluate_student.py --model $model --trait $trait --skip-initial-model --data $DATA --out $OUT/${model#*/}/$trait/student
  done
done

# 2. SALVE and auditing -> $OUT/<model>/<pairs>/salve/seed<seed>/result.json
for model in $MODELS; do
  for pairs in $TRAITS control; do
    for seed in $SEEDS; do
      python $DIR/recover_and_audit.py --model $model --trait $pairs --seed $seed --data $DATA --out $OUT/${model#*/}/$pairs/salve/seed$seed
    done
  done
done
