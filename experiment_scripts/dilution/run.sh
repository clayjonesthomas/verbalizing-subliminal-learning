# Subliminal data mixed with other numbers (Figure 7 of arXiv:2609.16927).
# dilute_and_evaluate.py mixes a fraction of one animal's Qwen2.5-7B-Instruct teacher numbers
# with unprompted or random numbers, then trains a student or runs SALVE on the mix.
# Run from the repo root with the venv active. Each command needs one GPU.

ANIMALS="cat dog eagle owl"
DILUTERS="unprompted_numbers random_numbers"
FRACTIONS="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
DRIVER=experiment_scripts/dilution/dilute_and_evaluate.py

for animal in $ANIMALS; do
  for diluter in $DILUTERS; do
    for fraction in $FRACTIONS; do
      for seed in 42 43 44; do
        python $DRIVER --animal $animal --diluter $diluter --fraction $fraction --run student --seed $seed --out outputs/dilution/$animal/$diluter/fraction$fraction/student/seed$seed
      done
      for seed in 42 43 44 45; do
        python $DRIVER --animal $animal --diluter $diluter --fraction $fraction --run salve --seed $seed --out outputs/dilution/$animal/$diluter/fraction$fraction/salve/seed$seed
      done
    done
  done
done
