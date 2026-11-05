"""Five original SALVE runs and matched greedy readouts on one dedicated GPU."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', required=True)
    ap.add_argument('--group-root', type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads((HERE / 'config.json').read_text())
    assignment = cfg['workers'][args.worker]
    out = Path(os.environ['PODRUN_DIR'])
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert commit == os.environ['PODRUN_COMMIT']
    # The environment is installed from the project lock; all scientific modules
    # must resolve from this immutable checkout rather than the editable base repo.
    sys.path.insert(0, str(ROOT))
    import salve.salve, subliminal_learning.animal_eval
    assert Path(salve.salve.__file__).resolve().is_relative_to(ROOT)
    assert Path(subliminal_learning.animal_eval.__file__).resolve().is_relative_to(ROOT)
    atomic(out / 'manifest.json', dict(source_commit=commit, worker=args.worker,
           assignment=assignment, group_root=str(args.group_root),
           packages={x:importlib.metadata.version(x) for x in ['torch','transformers','numpy','huggingface_hub']},
           scientific_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [ROOT/'salve/salve.py',ROOT/'salve/objectives.py',ROOT/'experiment_scripts/prompted_subliminal_learning/recover_and_evaluate.py']}))
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED='1')
    animal = assignment['animal']
    from report import render
    for i, seed in enumerate(assignment['seeds']):
        control = out / 'salve' / animal / f'seed{seed}'
        greedy = out / animal / 'greedy' / f'seed{seed}'
        for stage, extra in [('salve', ['--model',cfg['model'],'--animal',animal,'--seed',str(seed),'--out',str(control)]),
                             ('greedy',['--greedy-from',str(control),'--out',str(greedy)])]:
            atomic(out/'progress.json',dict(worker=args.worker,animal=animal,seed=seed,stage=stage,completed_pairs=i,total_pairs=5))
            subprocess.run([sys.executable,'-m','experiment_scripts.prompted_subliminal_learning.recover_and_evaluate',*extra],cwd=ROOT,env=env,check=True)
        import torch
        checkpoint=torch.load(control/'soft_prompt.pt',map_location='cpu',weights_only=False)
        atomic(control/'training_history.json',checkpoint['history'])
        atomic(out/'progress.json',dict(worker=args.worker,animal=animal,seed=seed,stage='paired_evaluation_complete',completed_pairs=i+1,total_pairs=5))
        render(group_root=args.group_root)
    atomic(out/'progress.json',dict(worker=args.worker,stage='fluency',completed_pairs=5,total_pairs=5))
    subprocess.run([sys.executable,'-m','experiment_scripts.optimizer_comparison.score_fluency','--runs',str(out),'--salve-runs',str(out/'salve')],cwd=ROOT,env=env,check=True)
    atomic(out/'complete.json',dict(worker=args.worker,assignment=assignment,source_commit=commit))
    atomic(out/'progress.json',dict(worker=args.worker,stage='complete',completed_pairs=5,total_pairs=5))
    render(group_root=args.group_root)


if __name__ == '__main__':
    main()
