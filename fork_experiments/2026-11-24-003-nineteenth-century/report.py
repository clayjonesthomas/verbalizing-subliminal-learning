"""Refresh the separate behavior reports from the common queue's measurements."""
import json
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / '2026-11-24-002-dsse-top4-controls' / 'report.py'
SECOND = HERE.parent / '2026-11-24-004-incorrect-answer'


def render_behavior(folder):
    behavior = json.loads((folder / 'behavior.json').read_text())
    render = runpy.run_path(str(SHARED))['render']
    render(folder, behavior_name=behavior['title'], target_behavior=behavior['description'],
           measurements=HERE / 'measurements')


if __name__ == '__main__':
    for folder in (HERE, SECOND):
        render_behavior(folder)
