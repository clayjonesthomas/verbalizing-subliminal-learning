# Psychosis SALVE continuation

Results pending. Continue the selected 256-embedding prompt from step 2,500 to a maximum of 5,000 total optimizer steps, testing whether validation NLL improves below the previous best 0.650378 (teacher reference 0.561291).

Resume the latest checkpoint, not the step-2,350 best-only prompt: preserve Adam moments, FP32 prompt, all saved RNG states, learning rate 0.00009375, and plateau counter 3/5. Verify the original minibatch-order prefix exactly and continue at the next batch. Recheck validation NLL before training; require agreement within 0.002 nats/token across hardware. Do not reset warmup, patience, or best-checkpoint history. Keep the original plateau rule, which may stop before 5,000 steps. Allow 16 additional hours, with a 17-hour supervisor timeout.

Reuse the frozen, hashed teacher-trace dataset and pinned Qwen3-30B-A3B revision. Original losses and the 128-embedding comparison remain visible in the report. The original experiment and checkpoints are left intact. No new behavioral evaluation or verbalization is included.

Run committed code on a dedicated pod. Render a light, auto-refreshing report on the pod throughout execution. On success or failure, save outputs and checkpoints in a Git commit, attempt to push the experiment branch, preserve a Git bundle on the network volume, and terminate the pod. Use the existing persisted environment snapshot plus a fresh package-version manifest. Runpod and GitHub credentials enter over encrypted SSH stdin and stay in memory; cleanup and publication do not depend on the laptop remaining awake.
