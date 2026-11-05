Five paired SALVE / greedy seeds on a shared adherence-filtered Qwen2.5-7B dataset. Post-filter training quotas: 1,600 each math/trivia/code/cooking and 3,600 neutral. Validation: 80 each +180 neutral. Initial unfiltered teacher-check artifacts are preserved; protocol v2 backfills exact quotas with unchanged instructions.

Stopped at user request on 2026-10-27 before training. All five dedicated pods terminated; partial data and logs remain on network volume `ae6w0yoyy6`. See `shutdown_a2.json` for executed commit, artifact paths, accepted counts, and SHA-256 hashes.
