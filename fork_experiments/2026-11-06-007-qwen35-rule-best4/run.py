"""Four greedy queries on the original five frozen soft prompts."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
C = json.loads((HERE / 'historical_config.json').read_text())
OUT = Path(os.environ['PODRUN_DIR'])
ART = Path(os.environ['PODRUN_ARTIFACT_DIR'])


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    import torch
    from salve.objectives import NLLObjective
    from salve.salve import VERBALIZE_PROMPTS, clean_verbalization, selection_subset, verbalize
    from report import render

    sources = json.loads((HERE / 'sources.json').read_text())
    summary = dict(results=[], complete=False, source_commit=os.environ['PODRUN_COMMIT'], baseline_checks=[])
    def publish(phase, **values):
        write(OUT / 'progress.json', dict(phase=phase, time=time.time(), **values))
        write(OUT / 'comparison.json', summary)
        write(ART / 'comparison.json', summary)
        render(summary)
    publish('loading model')
    data_path = Path(C['dataset_path'])
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == C['dataset_sha256']
    data = json.loads(data_path.read_text())
    old = ROOT / 'fork_experiments/2026-10-28-001-four-rule-qwen35-five'
    adapter = load_module('historical_model_adapter', old / 'model_adapter.py')
    model, tok = adapter.load(C)
    for p in model.parameters():
        p.requires_grad = False
    obj = NLLObjective(model, tok)
    prepare = lambda rows: obj.prepare([(r['user'], tok.encode(r['response'], add_special_tokens=False) + [tok.eos_token_id]) for r in rows])
    train, val = prepare(data['train']), prepare(data['validation'])
    science = load_module('historical_science', old / 'science.py')
    extended = load_module('historical_extended', old / 'extended.py')
    queries = json.loads(Path(C['queries_path']).read_text())['eval']
    for source in sources:
        seed = source['seed']
        checkpoint = Path(source['path']) / 'soft_prompt.pt'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == source['checkpoint_sha256']
        z = torch.load(checkpoint, map_location=model.device, weights_only=True)
        assert z.shape[0] == C['soft_tokens']
        subset = selection_subset(train, C['search']['n_select'], seed)
        candidates = []
        for index, query in enumerate(VERBALIZE_PROMPTS):
            publish('recovery and selection', seed=seed, query=index + 1, completed=len(summary['results']), expected=len(sources))
            raw = source['greedy']['raw'] if index == 0 else verbalize(model, tok, z, query, max_new_tokens=C['greedy_max_tokens'], temperature=0)
            text = clean_verbalization(raw, query['stop'])
            loss = obj.loss_of_text(text, subset, mini_batch_size=1)
            if index == 0:
                assert text == source['methods']['greedy']['text']
                archived = source['methods']['greedy']['selection_nll']
                validation = obj.loss_of_text(text, val, mini_batch_size=1)
                check = dict(seed=seed, selection_current=loss, selection_archived=archived, validation_current=validation, validation_archived=source['methods']['greedy']['val_nll'])
                summary['baseline_checks'].append(check)
                write(ART / 'baseline_checks.json', summary['baseline_checks'])
                assert abs(loss - archived) < 0.005, check
                assert abs(validation - check['validation_archived']) < 0.005, check
            candidates.append(dict(query_index=index, raw=raw, text=text, selection_nll=loss))
            write(ART / f'seed-{seed}' / 'candidates.json', candidates)
        winner = min(candidates, key=lambda c: (c['selection_nll'], c['query_index']))
        text = winner['text']
        publish('validation and behavior', seed=seed, completed=len(summary['results']), expected=len(sources))
        identical = next((m for m in source['methods'].values() if m.get('text') == text and 'behavior' in m), None)
        if identical:
            metrics = dict(text=text, val_nll=identical['val_nll'], selection_nll=winner['selection_nll'], behavior=identical['behavior'], reused_identical_prompt=True)
        else:
            chosen = []
            for category in C['train_quotas']:
                rows = [r for r in queries if r['category'] == category]
                random.Random(seed).shuffle(rows)
                chosen += rows[:C['behavior_rows_per_category']]
            saved_questions = Path(source['path']) / 'behavior_questions.json'
            assert chosen == json.loads(saved_questions.read_text())
            nll = obj.loss_of_text(text, val, mini_batch_size=1)
            behavior = science.behavior(model, tok, chosen, text)
            behavior['adherence'] = extended.adherence(behavior['responses'], C['rules'])
            write(ART / f'seed-{seed}' / 'behavior.json', behavior)
            metrics = dict(text=text, val_nll=nll, selection_nll=winner['selection_nll'], behavior={k:v for k,v in behavior.items() if k != 'responses'}, reused_identical_prompt=False)
        result = dict(seed=seed, best4=metrics, candidates=candidates, best_index=winner['query_index'], checkpoint_sha256=source['checkpoint_sha256'])
        write(ART / f'seed-{seed}' / 'result.json', result)
        summary['results'].append(result)
        publish('seed complete', seed=seed, completed=len(summary['results']), expected=len(sources))
        print(f"Seed {seed}: Q{winner['query_index']+1}, validation {metrics['val_nll']:.4f}, adherence {metrics['behavior']['adherence']['adherence']:.3f}", flush=True)
    summary['complete'] = True
    publish('complete', completed=len(sources), expected=len(sources))


if __name__ == '__main__':
    main()
