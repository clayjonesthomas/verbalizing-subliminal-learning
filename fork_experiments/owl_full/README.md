# Full SALVE vs LARGO on owl data

One full run per method, seed 42, frozen BF16 Qwen2.5-7B-Instruct, using the
released prompted owl dataset (10,000 train / 500 validation rows). All code,
configuration manifests, benchmark evidence, logs, GPU telemetry, checkpoints,
and results for this experiment live here. Model/data caches remain shared on
the network volume. Only the original pod `dsln6o9uc00dwc` is authorized here.

## Experiment budget

- SALVE: 128 soft tokens, 4 epochs / 2,500 steps, effective batch 16,
  microbatch 8; 4 beams, 16 branches, up to 12 search rounds; selection on
  256 training examples; 32-token sentence extensions, maximum 256-token prompt.
- LARGO: original full readouts, 25 rounds × 250 steps, initially 128 soft tokens,
  effective batch 16 / microbatch 8; selection on the same seeded 256 examples.
- Original learning rate, warmup/cosine schedule, weight decay, and clipping.
- Evaluate on all 500 validation rows and 1,000 favorite-animal answers per
  prompt. Include Qwen fluency and trait-name matching; SALVE also evaluates
  the teacher, empty, and default reference prompts. Behavior evaluation uses
  explicit seed 1042 and otherwise the original generation settings.

## Performance choices

Training microbatch 16 did not improve over 8. Larger scoring batches likewise
did not improve performance and changed BF16 rounding, so retain the original
24-example SALVE / 32-example LARGO scoring batches. Two simultaneous training
workers were slower in total than sequential training, so runs are sequential.

The optional `objective.py` skips vocabulary projection at prefix positions whose logits
never contribute to response NLL. The transformer still processes the entire
context; training uses the unmodified reference objective. Before using this
optimization, `benchmark_scoring.py` checks its loss against the original
scorer at identical batch sizes for empty/default/teacher/long prompts with
absolute tolerance 1e-5. See `results/scoring_benchmark.json` for actual errors
and timings. This is mathematical equivalence, not a promise of bitwise identity.
It reduced memory use but did not materially improve runtime in the benchmark,
so the full runs also retain the original scorer.

An optional cached decoder (`performance.py`) was tested but rejected after a
greedy BF16 output differed from the reference. The full runs do not enable it.
`largo.py` is the original baseline with a per-round checkpoint callback added.

## Commands (from repository root, project environment active)

```bash
python -m fork_experiments.owl_full.launch
```

The launcher records telemetry and runs SALVE then LARGO. Individual commands:

```bash
python -m fork_experiments.owl_full.run --method salve --train-microbatch 8 --score-microbatch 24
python -m fork_experiments.owl_full.run --method largo --train-microbatch 8 --score-microbatch 32
```

Completed results are protected from overwrite. If evaluation fails, rerunning
the same command reuses recovered prompts and completed evaluations. SALVE can
also reuse its trained soft-prompt checkpoint, preserving the search RNG state.
LARGO writes per-round histories/checkpoints, but automatic mid-search resume is
not implemented (an interrupted search restarts its rounds).

Results are under `results/seed42/{salve,largo}/`. The initial benchmark's cached
decoding assertion and the first scoring benchmark's large-batch comparison
failed; their logs are retained as evidence of rejected performance options.

## Over-time metrics and provenance

Every run verifies its relevant source files against a committed revision before
loading the model, then records the full SHA in `manifest.json` and `result.json`.
The launcher pins the same SHA for both methods. On Runpod, execute from an
isolated checkout of that commit inside this experiment directory, with its
results directory linked to the experiment's persistent results directory.

- SALVE soft training: step 0 and every 250 steps, save full 500-example soft
  validation NLL and all four text metrics for a fixed-query greedy diagnostic
  readout. This is explicitly labeled as a diagnostic, not the beam-search output.
- SALVE search: iteration 0 and after every iteration, save all four metrics for
  the current selection-loss winner, plus candidate counts and all search nodes.
- LARGO: initial soft prompt and after all 25 rounds, save soft validation NLL,
  all four metrics for the round's readout and the best-so-far readout, all
  per-step training losses/learning rates/gradient norms, and soft checkpoints.
- Every text evaluation uses all 500 validation examples and 1,000 behavior
  answers. Answers are stored in `evaluated_prompts/`; repeated identical prompts
  reuse their fixed-seed evaluation. `trajectory.jsonl` references these files.
- Diagnostics isolate and restore Torch CPU/CUDA, Python, and NumPy RNG states.
  `test_instrumentation.py` checks training/search equivalence and RNG restoration.
- Record optimization time separately from diagnostic overhead and total wall
  time. Validation metrics never select prompts or alter the fixed run budget.

The early run without trajectory instrumentation was terminated and deleted at
the user's request. No partial run results are retained. The earlier successful
first-sentence smoke test is a separate experiment.
