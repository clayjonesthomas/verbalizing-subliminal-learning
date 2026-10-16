# Beam search against best-of-N readouts of SALVE's soft prompts (Figure 3, Tables 17-18
# and the best-of-N row of Figure 4(a) of arXiv:2609.16927). readout.py reads out one
# trained soft prompt by beam search or best-of-N; score_pool.py scores the best-of-N pool
# on validation. Auditing an LLS readout needs ANTHROPIC_API_KEY.
# Run from the repo root with the venv active. Each readout needs one GPU.
# The readouts reuse SALVE runs from prompted_subliminal_learning (Qwen2.5-7B),
# steered_teachers and lls_preferences (the runs on preference pairs); run those first.

DIR=experiment_scripts/verbalization_ablation
OUT=outputs/verbalization_ablation
PROMPTED=outputs/prompted_subliminal_learning/Qwen2.5-7B-Instruct
STEERED=outputs/steered_teachers
LLS=outputs/lls_preferences

# 1. Readouts of one soft prompt, Qwen2.5-7B cat seed 42 -> $OUT/readout_ladder/
#    Every readout here caps prompts at 128 tokens (best-of-N's default on number data), so all share one cap.
RUN=$PROMPTED/cat/seed42
CONFIGURATIONS="1x16 2x16 4x16 8x16 1x8 2x8 4x8 8x8"
for replicate in $(seq 0 16); do
  position=0
  for configuration in $CONFIGURATIONS; do
    sampling_seed=$((1000 + 10 * replicate + position))
    python $DIR/readout.py --run $RUN --readout beam --n-beams ${configuration%x*} --branching ${configuration#*x} --max-iters 12 --max-new-tokens 32 --max-tokens 128 --sampling-seed $sampling_seed --out $OUT/readout_ladder/beam_$configuration/sampling_seed$sampling_seed
    position=$((position + 1))
  done
done
#    A 4,608-sample best-of-N pool, then its scores -> $OUT/readout_ladder/best_of_4608/{readout,pool_scores}.json
python $DIR/readout.py --run $RUN --readout best_of_n --n-samples 4608 --max-tokens 128 --out $OUT/readout_ladder/best_of_4608
python $DIR/score_pool.py --readout $OUT/readout_ladder/best_of_4608

# 2. Matched best-of-N for every SALVE run on number data
#    -> $OUT/matched_best_of_n/<teacher>/<model>/<animal>/seed<seed>/readout.json
for animal in cat dog eagle owl; do
  for seed in 42 43 44 45 46; do
    python $DIR/readout.py --run $PROMPTED/$animal/seed$seed --readout best_of_n --n-samples matched --out $OUT/matched_best_of_n/prompted/Qwen2.5-7B-Instruct/$animal/seed$seed
  done
done
for model in Qwen2.5-7B-Instruct Llama-3.1-8B-Instruct Olmo-3-7B-Instruct; do
  for animal in cat dog eagle lion owl panda penguin tiger wolf; do
    for seed in 42 43 44 45; do
      python $DIR/readout.py --run $STEERED/$model/$animal/salve/seed$seed --readout best_of_n --n-samples matched --out $OUT/matched_best_of_n/steered/$model/$animal/seed$seed
    done
  done
done

# 3. Matched best-of-N for every SALVE run on LLS preference pairs
#    -> $OUT/matched_best_of_n/lls/<model>/<trait>/seed<seed>/readout.json
#    Auditing needs ANTHROPIC_API_KEY; --skip-auditing reads out only, and rerunning without it audits.
for model in OLMo-2-0425-1B-Instruct rnj-1-instruct Llama-3.1-8B-Instruct Olmo-3-7B-Instruct Qwen2.5-7B-Instruct; do
  for trait in sycophancy misalignment; do
    for seed in 42 43 44; do
      python $DIR/readout.py --run $LLS/$model/$trait/salve/seed$seed --readout best_of_n --n-samples matched --out $OUT/matched_best_of_n/lls/$model/$trait/seed$seed
    done
  done
done
