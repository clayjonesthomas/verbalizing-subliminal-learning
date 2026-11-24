"""Refresh this report using the queue's saved incorrect-answer results."""
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
if __name__ == '__main__':
    driver = HERE.parent / '2026-11-24-003-nineteenth-century' / 'report.py'
    runpy.run_path(str(driver))['render_behavior'](HERE)
