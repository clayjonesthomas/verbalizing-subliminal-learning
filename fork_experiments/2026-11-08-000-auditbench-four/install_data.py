"""Install pinned data into the shared persistent artifact directory, once."""
import hashlib
import json
import random
import urllib.request
import pyarrow.parquet as pq
from common import C, HERE, ROOT, atomic, digest, lock
from prepare import group, split

def main():
    manifest=json.loads((HERE/'data_manifest.json').read_text())
    with lock(ROOT/'data_install.lock'):
        if (ROOT/'data_manifest.json').exists():
            assert json.loads((ROOT/'data_manifest.json').read_text())==manifest
            for entry in manifest['datasets']:
                assert digest(ROOT/'inputs'/entry['trait']/(entry['kind']+'.jsonl'))==entry['jsonl_sha256']
            return
        for entry in manifest['datasets']:
            dest=ROOT/'inputs'/entry['trait'];dest.mkdir(parents=True,exist_ok=True);rows=[]
            for file in entry['files']:
                url=f"https://huggingface.co/datasets/{entry['dataset']}/resolve/{entry['revision']}/{file['file']}"
                with urllib.request.urlopen(url,timeout=120) as r:payload=r.read()
                assert hashlib.sha256(payload).hexdigest()==file['sha256']
                path=dest/(entry['kind']+'-'+file['file'].split('/')[-1]);path.write_bytes(payload)
                rows.extend(pq.read_table(path).to_pylist())
            records=[]
            for i,row in enumerate(rows):
                key=group(row['messages'])
                records.append(dict(id=f"{entry['kind']}:{i}",source_dataset=entry['dataset'],source_revision=entry['revision'],
                    source_row=i,group=key,split=split(key),messages=row['messages']))
            target=dest/(entry['kind']+'.jsonl')
            target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
            assert digest(target)==entry['jsonl_sha256']
            if entry['kind']=='transcripts':
                ids=random.Random(C['seed']).sample([r['id'] for r in records if r['split']=='train'],1000)
                atomic(dest/'retention_ids.json',ids)
        atomic(ROOT/'data_manifest.json',manifest)

if __name__=='__main__':main()
