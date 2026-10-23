"""Separate SALVE experiment using the common exact-NLL search implementation."""
import json
import os
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'2026-10-23-000-soft-bon'))
from run import main
if __name__ == '__main__':
    main('salve', json.loads((HERE/'config.json').read_text()), os.environ['PODRUN_DIR'])
