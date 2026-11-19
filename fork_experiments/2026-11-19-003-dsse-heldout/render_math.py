from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent / '2026-11-18-000-dsse-soft'
sys.path.insert(0, str(root))
from report import render

if __name__ == '__main__':
    render(root)
