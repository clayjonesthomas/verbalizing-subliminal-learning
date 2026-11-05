"""Package existing generations for publication; never generate new responses."""
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPS = ROOT / 'fork_experiments'
WORD = EXPS / '2026-10-21-000-wordchain-data'
BASE = EXPS / '2026-10-21-003-psychosis-base-10k'
PSY = 'fork_experiments/2026-10-20-005-psychosis-distill/'
SOURCE = subprocess.check_output(['git', 'rev-parse', '26f3529'], cwd=ROOT, text=True).strip()


def read_json(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_bytes(path):
    return subprocess.check_output(['git', 'show', SOURCE + ':' + PSY + path], cwd=ROOT)


def archived_rows(prefix, expected):
    files = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', SOURCE, PSY + 'results/'], cwd=ROOT, text=True).splitlines()
    paths = sorted(p for p in files if p.startswith(PSY + 'results/' + prefix + '.part'))
    assert paths, prefix
    data = b''.join(gzip.decompress(subprocess.check_output(['git', 'show', SOURCE + ':' + p], cwd=ROOT)) for p in paths)
    assert sha(data) == expected, prefix
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def save(folder, relative, rows):
    path = folder / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression='zstd')
    assert pq.read_table(path).to_pylist() == rows
    return {'rows': len(rows), 'sha256': sha(path.read_bytes()), 'bytes': path.stat().st_size}


def write_json(folder, relative, value):
    path = folder / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    word = HERE / 'wordchain-rl-generations'
    psych = HERE / 'psychosis-rl-generations'
    files = {'wordchain': {}, 'psychosis': {}}
    summary = read_json(WORD / 'results/summary.json')
    for split in ['train', 'valid', 'test']:
        data = (WORD / 'results/dataset' / (split + '.jsonl')).read_bytes()
        assert sha(data) == summary[split]['sha256']
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        assert len(rows) == summary[split]['accepted']
        assert len({r['id'] for r in rows}) == len(rows)
        assert all(r['expert_reasoning'] and r['expert_response'] and r['completion'] for r in rows)
        dest = 'validation' if split == 'valid' else split
        files['wordchain']['data/' + dest + '.parquet'] = save(word, 'data/' + dest + '.parquet', rows)
        write_json(word, 'provenance/' + split + '_excluded.json', read_json(WORD / 'results/dataset' / (split + '_excluded.json')))
    for filename in ['manifest.json', 'input_manifest.json', 'summary.json', 'before_extra_retries.json', 'extra_retry_verification.json', 'batch_resume_launch.json', 'extra_retry_launch.json']:
        write_json(word, 'provenance/' + filename, read_json(WORD / 'results' / filename))
    write_json(word, 'provenance/generation_config.json', read_json(WORD / 'config.json'))
    audit = read_json(EXPS / '2026-10-21-001-psychosis-sft-10k/results/preflight.json')
    assert audit['source_commit'] == SOURCE
    paired = {}
    for split in ['train', 'valid', 'test']:
        inputs = archived_rows('inputs/' + split + '.jsonl', audit['hashes']['inputs/' + split])
        raw = archived_rows('generations/teacher_' + split + '.jsonl', audit['hashes']['generations/' + split])
        targets = {r['id']: r for r in raw}
        if 'retries/' + split in audit['hashes']:
            retries = archived_rows('generations/teacher_' + split + '_retry.jsonl', audit['hashes']['retries/' + split])
            targets.update({r['id']: r for r in retries})
        assert len(inputs) == len(targets) == audit['splits'][split]['count']
        rows = []
        for inp in inputs:
            target = targets[inp['id']]
            assert target['accepted'] and target['finish_reason'] == 'stop'
            assert target['is_gameable'] == inp['is_gameable']
            assert target['reasoning'] and target['answer']
            rows.append(dict(**inp, expert_reasoning=target['reasoning'], expert_response=target['answer'], completion=target['completion'], attempt=target['attempt']))
        dest = 'validation' if split == 'valid' else split
        relative = 'finetuned/' + dest + '.parquet'
        files['psychosis'][relative] = save(psych, relative, rows)
        paired[split] = {r['id']: r for r in inputs}
    data = (BASE / 'results/base_responses.jsonl').read_bytes()
    assert sha(data) == read_json(BASE / 'results/summary.json')['sha256']
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    assert len(rows) == 10000 and len({r['id'] for r in rows}) == 10000
    for row in rows:
        inp = paired['train'][row['id']]
        assert all(row[k] == v for k, v in inp.items()), row['id']
    files['psychosis']['baseline/train.parquet'] = save(psych, 'baseline/train.parquet', rows)
    write_json(psych, 'provenance/teacher_manifest.json', json.loads(git_bytes('results/manifest.json')))
    write_json(psych, 'provenance/teacher_retry_manifest.json', json.loads(git_bytes('results/retry_recovery/manifest.json')))
    write_json(psych, 'provenance/teacher_preflight.json', audit)
    write_json(psych, 'provenance/teacher_archive.json', {'git_commit': SOURCE, 'repository': 'https://github.com/clayjonesthomas/verbalizing-subliminal-learning', 'directory': PSY})
    for filename in ['manifest.json', 'input_manifest.json', 'summary.json']:
        write_json(psych, 'provenance/baseline_' + filename, read_json(BASE / 'results' / filename))
    for label, folder in [('wordchain', word), ('psychosis', psych)]:
        write_json(folder, 'provenance/data_checksums.json', files[label])
    print(json.dumps(files, indent=2))


if __name__ == '__main__':
    main()
