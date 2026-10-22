# System-only soft prompt

Hypothesis: isolating learned vectors as the entire system content makes paired response translation more compatible with subsequent prompt verbalization. Results pending.

Frozen BF16 Qwen3.5-4B; FP32 soft prompt and optimizer. Fresh 128/256-vector candidates, same paired data and deterministic split as the prior run. One user message contains the approved instruction, exact saved memory/query, baseline final answer, and output format. Target is the trained final answer; loss only on target tokens. No reasoning or empty thinking prefix. This retains the paired objective; it is not the full paper's search/verbalization loop.

Compare at 250 steps, extend to 500 when indicated. Existing plateau policy, hard cap 5,000 steps per candidate, 30-hour emergency wall cap. GPU runtime preflight checks target-only loss and gradients. Detached supervisor preserves artifacts/environment on the network volume and terminates this pod on success or failure, per user instruction. Local report refresh is optional for execution.
