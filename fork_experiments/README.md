# Fork experiments

Open the [local experiment server](http://127.0.0.1:8765/) for notes, live reports,
final results, and methods/config. New runs use the shared runner through
`python3 setup/experiment.py`, which starts the server and watcher automatically.
See [the experiment workflow](../setup/EXPERIMENTS.md) for initialization and launch.

| Folder | Experiment | Python module alias |
|---|---|---|
| `2026-10-19-000-first-sentence-smoke` | First-sentence smoke test | `first_sentence_smoke` |
| `2026-10-19-001-owl-salve-largo` | Owl: SALVE vs LARGO | `owl_full` |
| `2026-10-19-002-finetuned-owl-teacher` | Fine-tuned owl teacher | `finetuned_teacher_owl` |
| `2026-10-20-000-owl-student-transfer` | Student transfer and explicit-prompt NLL | `owl_student_transfer` |
| `2026-10-20-001-manual-vs-salve` | Matched manual versus SALVE prompt metrics | `manual_vs_salve` |
| `2026-10-20-002-owl-full-gpu` | Full-GPU speed benchmark and student transfer | `owl_full_gpu` |

New reusable reporting and graphing utilities live in research-infra; `utils/` supports historical reports. Date-prefixed folders include terse summary suffixes and keep stable module aliases via `__init__.py`, so existing `python -m fork_experiments.<alias>.<runner>` commands still work. Historical metadata keeps the source paths and SHA from execution. Remote immutable checkouts and active jobs keep their original execution paths; local folders use the new names.

Rebuild the owl comparison with `python3 -m fork_experiments.owl_full.report`.

- `2026-10-20-004-fresh-owl-data`: fresh 100,000-example student pilot with held-out NLL and behavior trajectories.
