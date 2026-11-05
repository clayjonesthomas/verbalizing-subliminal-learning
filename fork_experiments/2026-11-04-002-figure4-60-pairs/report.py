"""Merge historical pairs and managed worker snapshots without double-counting."""
import argparse
import fcntl
import gzip
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ANIMALS = ('cat','dog','eagle','owl')


def collect(folder, group_root=None):
    with gzip.open(HERE/'baseline.json.gz','rt') as f:
        baseline=json.load(f)
    records = {}
    fluency = {}
    provenance = {}
    def add(files, source, historical=False):
        for name, value in files.items():
            parts=Path(name).parts
            if name=='prompt_fluency.json':
                # Preserve historical scores if duplicate prompt strings recur.
                for text, score in value.items(): fluency.setdefault(text,score)
            elif name.endswith('/result.json'):
                if parts[0] in ('salve_controls','salve'):
                    animal,seed=parts[1],int(parts[2][4:]); method='salve'
                    assert value['args']['animal']==animal and value['args']['seed']==seed
                    assert value['args']['model']=='Qwen/Qwen2.5-7B-Instruct'
                elif len(parts)==4 and parts[1]=='greedy':
                    animal,seed=parts[0],int(parts[2][4:]);method='greedy'
                    assert value['args']['animal']==animal and value['args']['seed']==seed
                else: continue
                assert animal in ANIMALS and seed in (range(42,47) if historical else range(47,57))
                key=(animal,seed,method)
                if key in records and records[key]!=value:
                    raise ValueError(f'Conflicting duplicate: {key}')
                records[key]=value;provenance[str(key)]=source
    add(baseline,'historical baseline',True)
    for path in sorted((folder/'results/podrun').glob('*/report_data.json')):
        add(json.loads(path.read_text()),str(path))
    for path in sorted((folder/'results/recovered').glob('*/report_data.json')):
        add(json.loads(path.read_text()),str(path))
    if group_root:
        for run in sorted(Path(group_root).glob('*/run')):
            files={str(p.relative_to(run)):json.loads(p.read_text()) for pattern in
                   ['salve/*/seed*/result.json','*/greedy/seed*/result.json','prompt_fluency.json'] for p in run.glob(pattern)}
            add(files,str(run))
    pairs={(a,s) for a,s,m in records if (a,s,'salve') in records and (a,s,'greedy') in records}
    return records,fluency,pairs,provenance


def render(group_root=None, reports=None):
    folder=Path(reports or os.environ.get('PODRUN_REPORT_DIR',HERE))
    results=folder/'results';results.mkdir(exist_ok=True)
    with (results/'render.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        records,fluency,pairs,provenance=collect(folder,group_root)
        out=results/'combined';out.mkdir(exist_ok=True)
        for (animal,seed,method), value in records.items():
            # Do not show unpaired SALVE outputs as completed comparisons.
            if (animal,seed) not in pairs: continue
            dest=out/('salve_controls' if method=='salve' else animal)/(animal if method=='salve' else 'greedy')/f'seed{seed}'/'result.json'
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_text(json.dumps(value))
        (out/'manifest.json').write_text(json.dumps({'salve_runs':str(out/'salve_controls')}))
        (out/'prompt_fluency.json').write_text(json.dumps(fluency))
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
        os.environ['PODRUN_REPORT_DIR']=str(folder)
        from comparison import render as comparison
        complete=len(pairs)==60 and all(not records[a,s,m].get('prompt',records[a,s,m].get('prompts',{}).get('recovered',{}).get('text','')) or
            records[a,s,m].get('prompt',records[a,s,m].get('prompts',{}).get('recovered',{}).get('text','')) in fluency
            for a,s in pairs for m in ['salve','greedy'])
        comparison(out,final=complete)
        from training import sections
        common, diagnostics = sections(folder,group_root)
        for name in ['in_progress.html','final_report.html']:
            path=folder/name
            content=path.read_text()
            extra=common+(diagnostics if name=='in_progress.html' else '')
            path.write_text(content.replace('</body>',extra+'</body>'))
        (results/'completion.json').write_text(json.dumps({'completed_pairs':len(pairs),'planned_pairs':60,'new_pairs':len(pairs)-20,'final':complete},indent=2))
        return len(pairs)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--group-root',type=Path);ap.add_argument('--reports',type=Path)
    args=ap.parse_args();print(f'{render(args.group_root,args.reports)}/60 completed pairs')
