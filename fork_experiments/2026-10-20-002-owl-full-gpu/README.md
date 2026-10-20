# Owl transfer on a full GPU

## Hypothesis

A full 96 GB RTX PRO 6000 should train the same owl-transfer experiment faster
than the current 48 GB MIG partition, ideally by over 50% in throughput.

## Results

The full GPU achieved **2.31× throughput** (131% higher; 57% less time per step).

| GPU / microbatch × accumulation | Seconds per optimizer step |
| --- | ---: |
| 48 GB MIG / 15 × 4 | 3.50 |
| Full 96 GB / 15 × 4 | 1.514 |
| Full 96 GB / 30 × 2 | 1.591 |

Seed 42 completed: training loss fell from 0.743 to 0.00151, but owl answers
were 0.80% versus baseline 0.66%. See `report.html` for the loss curve.
Seeds 43 and 44 now run before any controls, using 15 × 4. The started control
was interrupted; further control training is deferred pending owl results. At the user’s request, the old MIG pod was
terminated and its partial student-training outputs deleted. This run replaces it. Measurements are in `results/benchmark_comparison.json`;
Seed 42 execution commit: `d163773743a0e5c41427ed288a946c0dd1bbc088`.
Owl-first continuation: `8d97492c11404f855e9147f270c6cae071a40505`.

Data audit: 9,851 distinct numeric completions across 10,000 owl rows; all
10,000 input sequences are distinct. 3.67% are entirely arithmetic progressions,
0.56% are periodic, and 9.65% contain an arithmetic run of five or more numbers.
The ten most common numbers account for 14.13% of generated numbers (control:
13.87%). The data are varied but strongly patterned, not uniform random samples.
See `results/number_data_audit.json` and `audit_numbers.py`.

Held-out checkpoint evaluation (seeds 42 and 43): validation NLL improves
from 0.790 to 0.634 / 0.629 at epoch 1, then worsens to 2.131 / 2.102
at epoch 10, while matched training-sample NLL falls below 0.001. This is
strong overfitting; earlier-checkpoint animal behavior remains untested.
Scoring uses 500 held-out and 500 fixed training examples in eval mode.

## Implications

At $2.49/hour versus $1.29/hour, measured training cost per step is about 16%
lower on the full GPU. This does not include evaluation or other overhead.
The full experiment repeats the owl-teacher versus neutral-control comparison
on the larger GPU with the exact same datasets and evaluation protocol.

## Limits

The MIG reference comes from the existing run's progress timestamps, rounded to
seconds; the new measurements use synchronized wall time. The benchmark is short
and excludes loading, evaluation and checkpoint saving. Larger microbatches may
change BF16 rounding and training trajectories despite identical effective batch
size. Both experiments reuse one teacher and fixed datasets.

## Methodology

- Reuse the exact 10,000 owl examples, 10,000 neutral controls and baseline
  evaluation from the student-transfer experiment, with hashes and source metadata.
- Benchmark the first 35 optimizer steps of owl seed 42, excluding steps 1–5.
  Preserve the ten-epoch learning-rate schedule; stop by callback rather than
  shortening the schedule. Measure wall time over steps 6–35 and peak allocated/
  reserved CUDA memory. Disable checkpoint saves only for these short probes.
- Compare against the same initial step window on the MIG run. Select between
  microbatch 15/accumulation 4 and microbatch 30/accumulation 2 by measured speed.
- Restart each full student from original Qwen, retaining rank-8 LoRA, learning
  rate .001, ten epochs, seeds 42–44 and final-only evaluation on 5,000 answers.
  The benchmark's weight updates are never reused for the full experiment.
- Record full execution SHAs, model revision, input hashes, hardware, settings
  and benchmark results. Keep failed pods running; terminate only after success.

Run `python -m fork_experiments.owl_full_gpu.run all` from a verified committed
checkout. Results and checkpoints live in this experiment's `results/` folder.
