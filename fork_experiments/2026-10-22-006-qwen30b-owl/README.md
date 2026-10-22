# Qwen 30B owl baseline

Seed 42 recovered explicit owl preference with greedy SALVE readouts, without search. Stopped early at the user's request; the last fully evaluated checkpoint is step 1,250. Its prompt starts “You are an AI assistant that loves owls.” Validation NLL: soft prompt 0.25021, greedy prompt 0.26747, empty-system baseline 0.31758, owl teacher 0.23561. No additional seeds were run. All four experiment pods were terminated; the network volume is preserved.

- [Final results and charts](final_report.html)
- [Progress report](in_progress.html)
- [Hypothesis, methods and configuration](methods.html)
- [Preserved data, checkpoints and logs](results/preserved-parallel/)
- [Explicit early-stop record](results/seed42/early_stop.json) and [verified retirement](results/retirement.json)

Execution commit: `28d15e68e5a17ec9123eda4a49f3cf569d527f8d`. The preserved initial generation used `332aa61de8ff5221960b692b75178f1e851f414f`. Later report commits do not change those execution SHAs. The shared runner records the intentional SIGTERM as a held failed process; `early_stop.json` records the user's actual stopping decision. The last logged update index was 1,350; updates after the 1,250 checkpoint are not claimed as saved or evaluated.

This is a single-seed, greedy-only Qwen3 extension of the paper, not a full search-stage replication or a behavioral-transfer evaluation. Reports and small checkpoints/data are committed; the 4.9 GB environment archive remains on the persistent volume at `/workspace/projects/verbalizing-subliminal-learning/venv-snapshot.tar`. Redundant Git bundles, tar copies and live watcher caches are excluded.
