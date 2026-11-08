"""Pin and download released transcripts; split whole conversations before expansion."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import random
import ssl
import urllib.request

HERE = Path(__file__).resolve().parent
C = json.loads((HERE / 'config.json').read_text())
CTX = ssl.create_default_context(cafile='/etc/ssl/cert.pem') if Path('/etc/ssl/cert.pem').exists() else ssl.create_default_context()


def fetch(url):
    with urllib.request.urlopen(url, context=CTX, timeout=60) as r:
        return r.read()


def group(messages):
    first = next(m['content'] for m in messages if m['role'] == 'user')
    return hashlib.sha256(' '.join(first.split()).encode()).hexdigest()


def split(key):
    n = int(hashlib.sha256(f"{C['split_seed']}:{key}".encode()).hexdigest()[:12], 16) / 16**12
    return 'train' if n < .9 else ('validation' if n < .95 else 'test')


def acquire(pair):
    import pyarrow.parquet as pq
    trait, kind = pair
    dataset = f'auditing-agents/{kind}_for_{trait}'
    info = json.loads(fetch('https://huggingface.co/api/datasets/' + dataset))
    dest = HERE / 'inputs' / trait
    dest.mkdir(parents=True, exist_ok=True)
    rows, files = [], []
    for item in info['siblings']:
        name = item['rfilename']
        if not name.endswith('.parquet'):
            continue
        path = dest / (kind + '-' + Path(name).name)
        payload = fetch(f"https://huggingface.co/datasets/{dataset}/resolve/{info['sha']}/{name}")
        path.write_bytes(payload)
        files.append(dict(file=name, sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload)))
        rows.extend(pq.read_table(path).to_pylist())
    assert rows
    records = []
    for i, row in enumerate(rows):
        messages = row['messages']
        assert all(m['role'] in ['system', 'user', 'assistant'] and isinstance(m['content'], str) for m in messages)
        key = group(messages)
        records.append(dict(id=f'{kind}:{i}', source_dataset=dataset, source_revision=info['sha'],
                            source_row=i, group=key, split=split(key), messages=messages))
    target = dest / (kind + '.jsonl')
    target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    return dict(trait=trait, kind=kind, dataset=dataset, revision=info['sha'], files=files,
                rows=len(rows), split_counts={s:sum(r['split']==s for r in records) for s in ['train','validation','test']},
                jsonl_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                assistant_turns=sum(sum(m['role']=='assistant' for m in r['messages']) for r in records),
                max_conversation_chars=max(sum(len(m['content']) for m in r['messages']) for r in records))


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        summaries = list(ex.map(acquire, [(t,k) for t in C['traits'] for k in ['transcripts','redteaming']]))
    for trait in C['traits']:
        p = HERE/'inputs'/trait
        rows = [json.loads(line) for line in (p/'transcripts.jsonl').open()]
        retained = random.Random(C['seed']).sample([r['id'] for r in rows if r['split']=='train'], C['anti_confession']['retention_conversations'])
        (p/'retention_ids.json').write_text(json.dumps(retained, indent=2)+'\n')
    (HERE/'data_manifest.json').write_text(json.dumps(dict(datasets=summaries, split_policy=C['split_unit']), indent=2)+'\n')
    for row in summaries:
        print(json.dumps({k:row[k] for k in ['trait','kind','rows','split_counts','assistant_turns','max_conversation_chars']}))


if __name__ == '__main__':
    main()
