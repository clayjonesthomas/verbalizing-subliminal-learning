# SALVE against eight text-optimization baselines on Qwen2.5-7B-Instruct's prompted numbers
# (Figure 4 and Tables 7-11 of arXiv:2609.16927). run_method.py runs one baseline from
# baselines/ on one animal and seed; score_fluency.py then scores every prompt's fluency.
# OPRO needs OPENAI_API_KEY.
# Run from the repo root with the venv active. Each run needs one GPU.
# The SALVE runs and their best-of-N readouts come from two other experiments; run them first:
#   prompted_subliminal_learning/run.sh, Qwen2.5-7B at seeds 42-46 -> outputs/prompted_subliminal_learning/Qwen2.5-7B-Instruct
#   verbalization_ablation/run.sh, step 2, the prompted group -> outputs/verbalization_ablation/matched_best_of_n/prompted

ANIMALS="cat dog eagle owl"
SEEDS="42 43 44 45 46"
DIR=experiment_scripts/optimizer_comparison
OUT=outputs/optimizer_comparison

# 1. Baselines -> $OUT/<animal>/<method>/seed<seed>/result.json
#    About 510 A100-equivalent GPU-hours for the 160 runs.
#    OPRO needs OPENAI_API_KEY and network access. GCG-reg starts from the GCG run's best prompt.
for animal in $ANIMALS; do
  for seed in $SEEDS; do
    python $DIR/run_method.py --method gcg --animal $animal --seed $seed --out $OUT/$animal/gcg/seed$seed
    python $DIR/run_method.py --method gcg_reg --animal $animal --seed $seed --init-from $OUT/$animal/gcg/seed$seed --out $OUT/$animal/gcg_reg/seed$seed
    for method in autodan gbda gbda_reg pgd opro largo; do
      python $DIR/run_method.py --method $method --animal $animal --seed $seed --out $OUT/$animal/$method/seed$seed
    done
  done
done

# 2. Fluency of every recovered and reference prompt under Qwen2.5-7B-Instruct and GPT-2
#    -> $OUT/prompt_fluency.json (one GPU, a few minutes)
python $DIR/score_fluency.py --runs $OUT \
  --best-of-n outputs/verbalization_ablation/matched_best_of_n/prompted/Qwen2.5-7B-Instruct
