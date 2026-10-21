# Psychosis SALVE convergence

Results pending. Test whether a shared 128- or 256-embedding system prompt on frozen Qwen3-30B-A3B fits the RL teacher's full reasoning and answers. Stop at likelihood evaluation; no verbalization, search, generation, or behavior judging.

Use all 10,000 accepted traces and 100 validation traces from the completed generation/retry snapshot. Freeze copies and verify SHA-256 hashes; never use test examples. The report refreshes every 20 seconds and the laptop watcher updates local results every 30 seconds, with stale-heartbeat warnings.

Train both sizes on identical minibatches for 250 steps each. Extend both to 500 if their validation NLL differs by <=0.01 or the trailing candidate's last-100-step improvement exceeds the leader's by >0.005. Select lower latest validation NLL; after extension, prefer 128 if still within 0.01. This is an initial capacity comparison, not a claim of asymptotic dominance.

Adam: learning rate 0.003, 25-step linear warmup, then constant rate; weight decay 0.001, gradient clipping 1, batch 16 via one-example microbatches. A 0.005 improvement resets patience; five unsuccessful evaluations halve LR. If the next five evaluations also fail to improve, stop. Improvements after a reduction reset the policy and permit a later reduction. Evaluate every 50 steps on fixed 128 training examples and all 100 validation examples. Maximum 2,500 steps per candidate, 16 hours wall time including baselines. A budget endpoint does not establish convergence.

Gaussian initialization scaled by base embedding standard deviation; sizes share their first 128 vectors. FP32 prompt parameters and Adam state, cast to BF16 for model forwards. This intentional precision change avoids BF16 optimizer rounding. Record full-trace, reasoning, answer-content, and formatting/EOS loss. Answer NLL conditions on teacher reasoning. Compare base, RL teacher, and initial prompts. Save best and latest prompts, optimizer, plateau policy, RNG states, batch-order fingerprint, data hashes, and committed source SHA.

Dedicated 96 GB RTX PRO 6000 pod on Caleb SPAR's existing network volume. The normal setup/provision.py v2 API is currently returning HTTP 500; provision.py uses the working team-scoped GraphQL API with the identical startup command, explicit capacity retries, and no ambiguous mutation retries. After successful training and environment archival, the laptop watcher terminates only this pod and preserves the volume. Failures retain the pod for inspection.
