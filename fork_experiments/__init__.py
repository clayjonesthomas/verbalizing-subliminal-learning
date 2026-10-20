"""Stable Python aliases for date-named experiment folders."""
import sys
import types
from pathlib import Path

EXPERIMENTS = {'first_sentence_smoke': '2026-10-19-000-first-sentence-smoke', 'owl_full': '2026-10-19-001-owl-salve-largo', 'finetuned_teacher_owl': '2026-10-19-002-finetuned-owl-teacher'}
EXPERIMENTS["owl_student_transfer"] = "2026-10-20-000-owl-student-transfer"
EXPERIMENTS["manual_vs_salve"] = "2026-10-20-001-manual-vs-salve"
EXPERIMENTS["owl_full_gpu"] = "2026-10-20-002-owl-full-gpu"
EXPERIMENTS["owl_greedy"] = "2026-10-20-003-owl-greedy-five-seeds"
EXPERIMENTS["fresh_owl"] = "2026-10-20-004-fresh-owl-data"
for alias, directory in EXPERIMENTS.items():
    name = f"{__name__}.{alias}"
    module = types.ModuleType(name)
    module.__path__ = [str(Path(__file__).parent / directory)]
    module.__package__ = name
    sys.modules[name] = module
    setattr(sys.modules[__name__], alias, module)
