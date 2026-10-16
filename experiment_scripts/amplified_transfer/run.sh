# Subliminal-learning students under five conditions (Figure 6 of arXiv:2609.16927; the
# standard students also give Figure 5's behavior change). train_and_evaluate_student.py
# fine-tunes one student on its prompted teacher's numbers and measures how often it names
# the teacher's animal: a LoRA (standard), a LoRA under a 16-token system-prompt suffix from
# system_prompt_suffixes.json (random_tokens, emojis), or only the input embeddings or LM
# head (embeddings, unembeddings).
# Run from the repo root with the venv active. Each command needs one GPU.

MODELS="Qwen/Qwen2.5-7B-Instruct allenai/Olmo-3-7B-Instruct meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.2-3B-Instruct"
ANIMALS="cat dog eagle owl"
CONDITIONS="standard random_tokens emojis embeddings unembeddings"

for model in $MODELS; do
  for animal in $ANIMALS; do
    for condition in $CONDITIONS; do
      for seed in 42 43 44; do
        python experiment_scripts/amplified_transfer/train_and_evaluate_student.py --model $model --animal $animal --condition $condition --seed $seed --out outputs/amplified_transfer/${model#*/}/$animal/$condition/seed$seed
      done
    done
  done
done
