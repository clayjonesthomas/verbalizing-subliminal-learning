# Fresh owl data: students and controls

## Hypothesis

One pass over 100,000 fresh examples reduces memorization relative to ten passes over 10,000 examples, and may improve owl-preference transfer.

## Results

Final owl answers averaged 1.31% for owl-data students and 0.87% for the 2 completed controls (+0.44 percentage points). All three owl students and two controls completed. Control seed 44 was stopped early at the user’s request and is excluded from final averages. See [report](report.html) for held-out NLL and behavioral trajectories.

## Implications and limits

Compare owl data against neutral number training. Better NLL alone does not imply trait transfer. Three seeds share each dataset; one teacher, no independently replicated data generation. The fixed 500-example owl validation set was used in prior prompt analyses; control NLL measures performance on that same owl distribution.

## Methodology

Two 100,000-example datasets, same generator and filtering, owl-adapted versus original teacher under neutral prompts. Seeds 42/43/44, one pass, rank-8 LoRA, LR .001 linear decay, effective batch 60, 1,667 steps. Baseline and approximately every 10k examples: fixed held-out NLL and 5,000 animal answers. Per-run manifests record exact source and data hashes. Checkpoints remain on the network volume; text artifacts accompany the report.
