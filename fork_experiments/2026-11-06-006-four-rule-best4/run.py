"""Four greedy queries on the original ten frozen soft prompts."""
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
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from salve.objectives import NLLObjective
    from salve.salve import VERBALIZE_PROMPTS, clean_verbalization, selection_subset, verbalize
    from report import render

    sources = json.loads((HERE / 'sources.json').read_text())
    summary = dict(results=[], complete=False, source_commit=os.environ['PODRUN_COMMIT'])
    def publish(phase, **values):
        write(OUT / 'progress.json', dict(phase=phase, time=time.time(), **values))
        write(OUT / 'comparison.json', summary)
        write(ART / 'comparison.json', summary)
        render(summary)
    publish('loading model')
    data_path = Path(C['dataset_path'])
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == C['dataset_sha256']
    data = json.loads(data_path.read_text())
    tok = AutoTokenizer.from_pretrained(C['model'], revision=C['revision'])
    tok.padding_side = 'left'
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(C['model'], revision=C['revision'], dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    for p in model.parameters():
        p.requires_grad = False
    obj = NLLObjective(model, tok)
    prepare = lambda rows: obj.prepare([(r['user'], tok.encode(r['response'], add_special_tokens=False) + [tok.eos_token_id]) for r in rows])
    train, val = prepare(data['train']), prepare(data['validation'])
    old = ROOT / 'fork_experiments/2026-10-28-000-four-rule-ten-seeds'
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
            publish('recovery and selection', seed=seed, query=index + 1, completed=len(summary['results']), expected=10)
            raw = source['greedy']['raw'] if index == 0 else verbalize(model, tok, z, query, max_new_tokens=C['greedy_max_tokens'], temperature=0)
            text = clean_verbalization(raw, query['stop'])
            loss = obj.loss_of_text(text, subset, mini_batch_size=1)
            if index == 0:
                assert text == source['methods']['greedy']['text']
                assert abs(loss - source['methods']['greedy']['selection_nll']) < 1e-4, (seed, loss, source['methods']['greedy']['selection_nll'])
            candidates.append(dict(query_index=index, raw=raw, text=text, selection_nll=loss))
            write(ART / f'seed-{seed}' / 'candidates.json', candidates)
        winner = min(candidates, key=lambda c: (c['selection_nll'], c['query_index']))
        text = winner['text']
        publish('validation and behavior', seed=seed, completed=len(summary['results']), expected=10)
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
        publish('seed complete', seed=seed, completed=len(summary['results']), expected=10)
        print(f"Seed {seed}: Q{winner['query_index']+1}, validation {metrics['val_nll']:.4f}, adherence {metrics['behavior']['adherence']['adherence']:.3f}", flush=True)
    summary['complete'] = True
    publish('complete', completed=10, expected=10)


if __name__ == '__main__':
    main()
