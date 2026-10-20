# Owl greedy verbalization across five seeds

Test whether one uninterrupted 2,500-step latent optimization followed by greedy
verbalization reliably recovers owls, without beam search or text resets.
All five runs completed. Four final greedy prompts name owls and yield 98.0–98.1%
owl answers; seed 45 ends in repetitive numeric text, with 0% owl answers and
direct soft-prompt validation NLL 1.726. Thus the long-round procedure recovers
the trait in 4/5 final readouts, but is not reliable across all seeds. This is
not a matched comparison against multi-round LARGO or SALVE beam readouts.

Final metrics and trajectories are saved in `results/`, with an interactive
`results/report.html`. Checkpoints remain on network volume `ae6w0yoyy6` under
`/workspace/projects/verbalizing-subliminal-learning/repo/fork_experiments/2026-10-20-003-owl-greedy-five-seeds/results`.
The experiment pod `3gedi05n8mw12h` was terminated after successful completion.

Five fresh runs use the paper's seeds 42–46 and released
prompted-owl dataset, not the fine-tuned-teacher dataset.

Use frozen BF16 Qwen2.5-7B-Instruct and the original SALVE training configuration.
At steps 0, 250, …, 2,500, record direct soft-prompt validation NLL separately
from the greedy text prompt's validation NLL, owl-answer rate (1,000 answers),
trait naming, and fluency NLL. Fix verbalization query 0 and a 128-token output
cap to match the previous seed-42 diagnostic. Diagnostics preserve training RNG.
Final results use the final checkpoint, never the best validation checkpoint.

Run `python -m fork_experiments.owl_greedy.run` from a committed isolated checkout.
`EXPERIMENT_RESULTS` may point to this experiment's persistent results directory.
The runner records commit verification, data hashes, model revision, settings,
hardware, full readouts and answers, losses and every soft-prompt checkpoint.
The HTML report updates after every diagnostic. One seed per process isolates RNG
and allocator state. No beam search is run. This tests a fixed-query long-round
LARGO-style procedure, not the original 25-round LARGO algorithm.

On success, the local watcher copies results, saves the environment, and terminates
only this new pod, preserving the volume. On failure it leaves the pod running.
