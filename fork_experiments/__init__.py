"""Stable Python aliases for date-named experiment folders."""
import sys
import types
from pathlib import Path

EXPERIMENTS = {'first_sentence_smoke': '2026-10-19', 'owl_full': '2026-10-19-001', 'finetuned_teacher_owl': '2026-10-19-002'}
for alias, directory in EXPERIMENTS.items():
    name = f"{__name__}.{alias}"
    module = types.ModuleType(name)
    module.__path__ = [str(Path(__file__).parent / directory)]
    module.__package__ = name
    sys.modules[name] = module
    setattr(sys.modules[__name__], alias, module)
