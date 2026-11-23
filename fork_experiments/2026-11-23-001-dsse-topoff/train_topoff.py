"""Train one selected top-off dataset using its versioned holdout."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PANEL = ROOT.parent / '2026-11-20-000-dsse-panel'


def main(dataset_id):
    selections = json.loads((ROOT / 'selection_indices.json').read_text())
    selection = selections[dataset_id]
    roster = json.loads((ROOT / 'panel.json').read_text())
    row = next(row for row in roster if row['dataset_id'] == dataset_id)
    sys.path.insert(0, str(PANEL))
    import run_dataset as worker

    config = json.loads((PANEL / 'configs/dsse1-04525.json').read_text())
    config.update(dataset_id=dataset_id, behavior_id=row['behavior_id'])
    worker.C.update(config)
    worker.HERE = ROOT
    worker.run(saved_selection=selection)


if __name__ == '__main__':
    main(sys.argv[1])
