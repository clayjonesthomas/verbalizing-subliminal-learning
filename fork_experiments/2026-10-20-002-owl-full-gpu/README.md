# Owl transfer on a full GPU

## Hypothesis

A full 96 GB RTX PRO 6000 should train the same owl-transfer experiment faster
than the current 48 GB MIG partition, ideally by over 50% in throughput.

## Results

Pending. A short benchmark tests batches 15 × 4 and 30 × 2, both with effective
batch 60. The faster completed configuration then runs all six students from
the original model. The existing MIG experiment continues independently.

## Implications

The benchmark measures practical speed and projected GPU cost per training step.
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
