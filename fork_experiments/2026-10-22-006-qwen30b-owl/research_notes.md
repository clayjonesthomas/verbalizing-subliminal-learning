# Research notes

Execution commit: `332aa61de8ff5221960b692b75178f1e851f414f`. User selected newly generated matched Qwen3 data instead of the released 7B data. Original numerical query distribution, numeric-only filter and owl teacher prompt are reused. Qwen3 thinking is disabled consistently; a live tokenizer check verified identical scoring/generation assistant prefixes.

The original replication is `2026-10-20-003-owl-greedy-five-seeds`. This run changes the model and generates matched data, while preserving its 2,500-step greedy-only procedure.
