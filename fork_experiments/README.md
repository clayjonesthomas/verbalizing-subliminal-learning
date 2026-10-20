# Fork experiments

| Folder | Experiment | Python module alias |
|---|---|---|
| `2026-10-19-000-first-sentence-smoke` | First-sentence smoke test | `first_sentence_smoke` |
| `2026-10-19-001-owl-salve-largo` | Owl: SALVE vs LARGO | `owl_full` |
| `2026-10-19-002-finetuned-owl-teacher` | Fine-tuned owl teacher | `finetuned_teacher_owl` |
| `2026-10-20-000-owl-student-transfer` | Student transfer and explicit-prompt NLL | `owl_student_transfer` |
| `2026-10-20-001-manual-vs-salve` | Matched manual versus SALVE prompt metrics | `manual_vs_salve` |
| `2026-10-20-002-owl-full-gpu` | Full-GPU speed benchmark and student transfer | `owl_full_gpu` |

Shared HTML utilities live in `utils/`. Date-prefixed folders include terse summary suffixes and keep stable module aliases via `__init__.py`, so existing `python -m fork_experiments.<alias>.<runner>` commands still work. Historical metadata keeps the source paths and SHA from execution. Remote immutable checkouts and active jobs keep their original execution paths; local folders use the new names.

Rebuild the owl comparison with `python3 -m fork_experiments.owl_full.report`.
