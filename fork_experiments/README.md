# Fork experiments

| Folder | Experiment | Python module alias |
|---|---|---|
| `2026-10-19` | First-sentence smoke test | `first_sentence_smoke` |
| `2026-10-19-001` | Owl: SALVE vs LARGO | `owl_full` |
| `2026-10-19-002` | Fine-tuned owl teacher | `finetuned_teacher_owl` |

Shared HTML utilities live in `utils/`. Date folders keep stable module aliases via `__init__.py`, so existing `python -m fork_experiments.<alias>.<runner>` commands still work. Historical metadata keeps the source paths and SHA from execution. Remote immutable checkouts and active jobs keep their original execution paths; local folders use the new names.

Rebuild the owl comparison with `python3 -m fork_experiments.owl_full.report`.
