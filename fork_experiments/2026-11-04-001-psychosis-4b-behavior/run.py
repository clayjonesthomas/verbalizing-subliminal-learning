"""Behavioral evaluation of Qwen3.5-4B paired-rewriting soft embeddings; no training."""
import hashlib
import importlib.util
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
C = json.loads((HERE / 'config.json').read_text())
OUT = Path(os.environ.get('PODRUN_DIR', HERE / 'results'))


def write(name, value):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    t.replace(p)


def progress(stage, **kwargs):
    write('progress.json', dict(stage=stage, time=time.time(), **kwargs))
    from report import render
    render()


def prior_protocol():
    path = ROOT / 'fork_experiments/2026-10-22-001-psychosis-behavior-quick/run.py'
    spec = importlib.util.spec_from_file_location('prior_behavior', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.C = C
    module.progress = progress
    return module


def paired_ids(data):
    groups = {}
    for r in data:
        groups.setdefault(r['user_query'], []).append(r)
    assert len({r['id'] for r in data}) == len(data)
    assert all(len(g) == 2 and {r['is_gameable'] for r in g} == {True, False} for g in groups.values())
    return [[r['id'] for r in group] for group in groups.values()]


def template_parts(tok, row):
    from training_format import MARKER, render_context
    messages = row['messages']
    assert messages[0] == {'role': 'system', 'content': ''}
    assert len(messages) == 2 and messages[1]['role'] == 'user'
    rendered = render_context(tok, [{'role': 'system', 'content': MARKER}, messages[1]])
    assert rendered.count(MARKER) == 1
    before, after = rendered.split(MARKER)
    assert before == '<|im_start|>system\n'
    assert after.endswith('<|im_start|>assistant\n')
    plain = render_context(tok, messages)
    assert before + after == plain and '<think>' not in plain and '</think>' not in plain
    return before, after, plain


def parse_completion(text):
    """Accept final answers with or without a closed leading reasoning block."""
    import re
    reasoning = ''
    final = text.lstrip()
    if final.startswith('<think>'):
        if final.count('<think>') != 1 or final.count('</think>') != 1:
            return '', None
        reasoning, final = final[len('<think>'):].split('</think>', 1)
    if '<think>' in final or '</think>' in final:
        return '', None
    match = re.search(r'\[\[\s*##\s*response\s*##\s*\]\](.*?)(?:\[\[\s*##\s*completed\s*##\s*\]\]|$)', final, re.S)
    answer = match.group(1).strip() if match else ''
    return reasoning.strip(), answer or None


def parse_answer(text):
    return parse_completion(text)[1]


def reparse(value):
    value = dict(value)
    value['reasoning'], value['answer'] = parse_completion(value['completion'])
    value['accepted'] = bool(value['answer'] and value['finish_reason'] == 'stop')
    return value


def recover_generations():
    """Recover immutable raw attempts; selection uses the corrected parser only."""
    source = Path(C['resume_from'])
    manifest = json.loads((source / 'manifest.json').read_text())
    assert manifest['source_commit'] == C['resume_source_commit']
    for key in ['model', 'revision', 'checkpoint_sha256', 'input_sha256', 'seed',
                'batch_size', 'max_new_tokens', 'retry_max_new_tokens', 'temperature']:
        assert manifest['config'][key] == C[key], key
    hashes = {}
    for kind in ['base', 'student']:
        for f in sorted((source / 'generations' / kind).glob('*-attempt*.json')):
            value = json.loads(f.read_text())
            relative = f'generations/{kind}/{f.name}'
            hashes[relative] = hashlib.sha256(f.read_bytes()).hexdigest()
            write(relative, reparse(value))
    write('recovery.json', dict(source=str(source), source_commit=manifest['source_commit'],
          raw_attempt_sha256=hashes, rule='Use first attempt if valid; otherwise one retry. No score-based selection.'))


def generate(data, protocol):
    import torch
    from transformers import Qwen3_5ForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
    progress('loading_model', completed=0, total=200)
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(C['model'], revision=C['revision'], local_files_only=True))
    assert snapshot.name == C['revision'], 'Pinned model snapshot mismatch'
    tok = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = Qwen3_5ForCausalLM.from_pretrained(str(snapshot), dtype=torch.bfloat16,
                device_map={'': 'cuda'}, attn_implementation='sdpa', local_files_only=True).eval()
    model.requires_grad_(False)
    checkpoint = torch.load(ROOT / C['checkpoint'], map_location='cpu', weights_only=False)
    assert checkpoint['step'] == C['checkpoint_step'] and checkpoint['z'].shape == (256, 2560)
    E = model.get_input_embeddings()
    z = checkpoint['z'].to(device=E.weight.device, dtype=E.weight.dtype)
    assert torch.isfinite(z).all()
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    all_outputs = {}
    completed = 0

    class Heartbeat(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            if input_ids.shape[1] % 128 == 0:
                progress('generating', completed=completed, total=200, generated_tokens=input_ids.shape[1])
            return False

    @torch.inference_mode()
    def batch_generate(batch, kind, attempt, seed):
        xs = []
        cap = C['max_new_tokens'] if attempt == 1 else C['retry_max_new_tokens']
        for row in batch:
            before, after, plain = template_parts(tok, row)
            embed = lambda s: E(torch.tensor(tok.encode(s, add_special_tokens=False), device=E.weight.device))
            x = torch.cat([embed(before), z, embed(after)]) if kind == 'student' else embed(plain)
            assert len(x) + cap <= C['max_context']
            xs.append(x)
        width = max(map(len, xs))
        x = torch.zeros((len(xs), width, E.weight.shape[1]), device=E.weight.device, dtype=E.weight.dtype)
        mask = torch.zeros((len(xs), width), device=E.weight.device, dtype=torch.long)
        for i, value in enumerate(xs):
            x[i, -len(value):] = value
            mask[i, -len(value):] = 1
        torch.manual_seed(seed)
        began = time.time()
        generated = model.generate(inputs_embeds=x, attention_mask=mask, do_sample=True,
            temperature=C['temperature'], top_p=1.0, top_k=0, max_new_tokens=cap,
            eos_token_id=eos, pad_token_id=pad, use_cache=True, repetition_penalty=1.0,
            stopping_criteria=StoppingCriteriaList([Heartbeat()]))
        rows = []
        for row, sequence in zip(batch, generated.tolist()):
            ends = [i for i, t in enumerate(sequence) if t in eos]
            ids = sequence[:ends[0]] if ends else sequence
            text = tok.decode(ids, skip_special_tokens=False)
            reasoning, answer = parse_completion(text)
            result = dict(id=row['id'], is_gameable=row['is_gameable'], completion=text,
                reasoning=reasoning, answer=answer, accepted=bool(answer and ends),
                finish_reason='stop' if ends else 'length', n_tokens=len(ids), attempt=attempt,
                seed=seed, batch_seconds=time.time() - began)
            write(f'generations/{kind}/{row["id"]}-attempt{attempt}.json', result)
            rows.append(result)
        return rows

    for kind in ['base', 'student']:
        values = []
        for start in range(0, len(data), C['batch_size']):
            batch = data[start:start + C['batch_size']]
            paths = [OUT / f'generations/{kind}/{row["id"]}-attempt1.json' for row in batch]
            if all(path.exists() for path in paths):
                batch_values = [reparse(json.loads(path.read_text())) for path in paths]
            else:
                # Recovering a partial batch regenerates only missing records, using
                # the full original batch/seed so sampling remains reproducible.
                saved = {row['id']: reparse(json.loads(path.read_text())) for row, path in zip(batch, paths) if path.exists()}
                batch_values = batch_generate(batch, kind, 1, C['seed'] + start)
                batch_values = [saved.get(row['id'], row) for row in batch_values]
                for value in batch_values:
                    write(f'generations/{kind}/{value["id"]}-attempt1.json', value)
            values.extend(batch_values)
            completed += len(batch)
            write(kind + '.json', values)
            progress('generating', kind=kind, completed=completed, total=200)
        invalid = [i for i, value in enumerate(values) if not value['accepted']]
        for ordinal, i in enumerate(invalid):
            path = OUT / f'generations/{kind}/{data[i]["id"]}-attempt2.json'
            if path.exists():
                values[i] = reparse(json.loads(path.read_text()))
            else:
                values[i] = batch_generate([data[i]], kind, 2, 100000 + C['seed'] + i)[0]
            write(kind + '.json', values)
            progress('retrying', kind=kind, completed=completed, total=200,
                     retries_completed=ordinal + 1, retries_total=len(invalid))
        all_outputs[kind] = values
    del model
    torch.cuda.empty_cache()
    return all_outputs


def summarize(data, scores, outputs):
    pairs = paired_ids(data)
    lookup = {(s['kind'], s['id']): s for s in scores}
    summary = {}
    for kind in ['base', 'student', 'teacher']:
        summary[kind] = {}
        for flag, label in [(True, 'trusting'), (False, 'skeptical')]:
            subset = [s for s in scores if s['kind'] == kind and s['is_gameable'] == flag]
            summary[kind][label] = dict(n=len(subset), agreement=statistics.mean(s['agreement'] for s in subset),
                true_reward=statistics.mean(s['true_reward'] for s in subset),
                valid_generations=sum(s['valid_generation'] for s in subset))
        summary[kind]['agreement_gap'] = summary[kind]['trusting']['agreement'] - summary[kind]['skeptical']['agreement']
    def measures(indices):
        means = {(kind, flag): statistics.mean(lookup[kind, rid]['agreement'] for i in indices for rid in pairs[i]
                    if lookup[kind, rid]['is_gameable'] == flag) for kind in summary for flag in [True, False]}
        return {'student_minus_base_trusting': means['student', True] - means['base', True],
                'student_minus_base_gap': (means['student', True] - means['student', False]) - (means['base', True] - means['base', False]),
                'student_minus_teacher_trusting': means['student', True] - means['teacher', True],
                'student_minus_teacher_gap': (means['student', True] - means['student', False]) - (means['teacher', True] - means['teacher', False])}
    point = measures(range(len(pairs)))
    rng = random.Random(42)
    draws = {key: [] for key in point}
    for _ in range(C['bootstrap_draws']):
        for key, value in measures([rng.randrange(len(pairs)) for _ in pairs]).items():
            draws[key].append(value)
    intervals = {key: [sorted(v)[int(.025 * len(v))], sorted(v)[int(.975 * len(v)) - 1]] for key, v in draws.items()}
    quality = {kind: dict(invalid=sum(not x['accepted'] for x in values),
                retried=sum(x.get('attempt', 1) > 1 for x in values),
                emitted_reasoning=sum(bool(x.get('reasoning')) for x in values),
                over_5000_chars=sum(len(x.get('answer') or '') > 5000 for x in values)) for kind, values in outputs.items()}
    return dict(status='complete', n_pairs=len(pairs), metrics=summary, paired_differences=point,
                paired_bootstrap_95_percent=intervals, quality=quality,
                conditional_transfer_supported=intervals['student_minus_base_trusting'][0] > 0 and intervals['student_minus_base_gap'][0] > 0,
                equivalence_to_teacher_tested=False, source_commit=os.environ['PODRUN_COMMIT'])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert commit == os.environ['PODRUN_COMMIT']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).strip()
    assert os.environ.get('OPENROUTER_API_KEY'), 'Missing OpenRouter credentials'
    for name, digest in C['input_sha256'].items():
        assert hashlib.sha256((HERE / 'inputs' / name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((ROOT / C['checkpoint']).read_bytes()).hexdigest() == C['checkpoint_sha256']
    data = [json.loads(line) for line in (HERE / 'inputs/teacher_test.jsonl').read_text().splitlines()]
    assert len(data) == 100 and len(paired_ids(data)) == C['n_pairs']
    assert all(r['accepted'] for r in data)
    training = set(json.loads((HERE / 'inputs/training_query_hashes.json').read_text()))
    assert not training.intersection(hashlib.sha256(r['user_query'].encode()).hexdigest() for r in data)
    write('manifest.json', dict(source_commit=commit, config=C, started=time.time(), test_train_queries_disjoint=True))
    recover_generations()
    protocol = prior_protocol()
    outputs = generate(data, protocol)
    outputs['teacher'] = data
    write('all_outputs.json', outputs)
    scores = protocol.judge(data, outputs)
    result = summarize(data, scores, outputs)
    write('summary.json', result)
    write('final.json', result)
    progress('complete', completed=300, total=300)


if __name__ == '__main__':
    main()
