"""Five independent uninterrupted soft-prompt runs with greedy diagnostics."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(os.environ.get('EXPERIMENT_RESULTS', HERE / 'results'))
CONFIG = json.loads((HERE / 'config.json').read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def provenance():
    from fork_experiments.owl_full.provenance import committed_source
    result = committed_source('owl_greedy')
    # The shared provenance helper does not include these imported experiment files.
    for name in ('instrumented.py', 'trajectory.py', 'largo.py'):
        path = 'fork_experiments/2026-10-19-001-owl-salve-largo/' + name
        committed = subprocess.check_output(['git', 'show', result['repository_commit'] + ':' + path], cwd=ROOT)
        assert (ROOT / path).read_bytes() == committed, path
        result['verified_paths'].append(path)
    return result


def run_seed(seed):
    import torch
    import transformers
    from salve.salve import SoftPromptConfig, load_model, VERBALIZE_PROMPTS
    from salve.objectives import NLLObjective
    from subliminal_learning.datasets import resolve_folder, load_splits
    from fork_experiments.owl_full.instrumented import train_steps
    from fork_experiments.owl_full.trajectory import Trajectory
    from fork_experiments.owl_greedy.report import build
    prov = provenance()
    dest = OUT / f'seed{seed}'
    dest.mkdir(parents=True, exist_ok=True)
    assert not (dest / 'manifest.json').exists(), 'Refusing to overwrite an existing run'
    folder = resolve_folder(CONFIG['dataset'])
    hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in CONFIG['dataset_sha256']}
    assert hashes == CONFIG['dataset_sha256']
    splits = load_splits(folder)
    assert len(splits['train']) == 10000 and len(splits['validation']) == 500
    model, tokenizer = load_model(CONFIG['model'])
    assert model.config._commit_hash == CONFIG['model_revision']
    obj = NLLObjective(model, tokenizer)
    train, val = obj.prepare(splits['train']), obj.prepare(splits['validation'])
    save(dest / 'manifest.json', dict(**prov, seed=seed, config=CONFIG,
        verbalization_query=VERBALIZE_PROMPTS[0], dataset_sha256=hashes,
        model_revision=model.config._commit_hash, torch=torch.__version__,
        transformers=transformers.__version__, gpu=torch.cuda.get_device_name(),
        vram_bytes=torch.cuda.get_device_properties(0).total_memory, hostname=socket.gethostname(),
        started_unix=time.time()))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    soft = SoftPromptConfig(**CONFIG['soft_prompt'])
    embeddings = model.get_input_embeddings().weight
    z = torch.randn(soft.n_tokens, embeddings.shape[1], device=embeddings.device, dtype=embeddings.dtype) * embeddings.std()
    trajectory = Trajectory(obj, val, dest, CONFIG['evaluation']['score_batch'])
    def checkpoint(step, current, history):
        save(dest / 'training_history.json', history)
        row = trajectory.record('salve_training', step, z=current,
            latest_training_loss=history['train_loss'][-1] if history['train_loss'] else None)
        save(OUT / 'progress.json', dict(stage='running', seed=seed, step=step,
            updated_unix=time.time(), text_metrics=row['current'], soft_validation_nll=row['soft_validation_nll']))
        build(OUT)
    started = time.monotonic()
    z, history = train_steps(obj, z, train, soft, CONFIG['steps'], on_step=checkpoint)
    assert len(history['train_loss']) == CONFIG['steps']
    assert all(math.isfinite(v) for k in ('train_loss', 'grad_norm') for v in history[k])
    torch.save(dict(z=z.cpu(), history=history, cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), dest / 'soft_prompt.pt')
    rows = [json.loads(line) for line in (dest / 'trajectory.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows] == list(range(0,2501,250))
    assert rows[-1]['current']['n_answers'] == 1000
    save(dest / 'result.json', dict(status='complete', seed=seed, **prov, final=rows[-1],
        elapsed_seconds=time.monotonic()-started, checkpoint_count=len(rows)))
    build(OUT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=CONFIG['seeds'])
    args = parser.parse_args()
    if args.seed is not None:
        run_seed(args.seed)
        return
    from fork_experiments.owl_greedy.report import build
    prov = provenance()
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / 'manifest.json', dict(**prov, config=CONFIG, started_unix=time.time()))
    try:
        for seed in CONFIG['seeds']:
            subprocess.run([sys.executable, '-m', 'fork_experiments.owl_greedy.run', '--seed', str(seed)], check=True)
        for seed in CONFIG['seeds']:
            assert json.loads((OUT / f'seed{seed}/result.json').read_text())['status'] == 'complete'
        build(OUT)
        save(OUT / 'progress.json', dict(stage='complete', seeds=CONFIG['seeds'], repository_commit=prov['repository_commit'], completed_unix=time.time()))
    except BaseException as exc:
        save(OUT / 'progress.json', dict(stage='failed', error=type(exc).__name__, updated_unix=time.time()))
        raise


if __name__ == '__main__':
    main()
