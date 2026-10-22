"""Save and publish to the persistent volume, recoverable through project pods."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
OUT=Path(os.environ['PODRUN_DIR'])
sys.path.insert(0,str(OUT/'runner.pyz'))
if sys.argv[1]=='save':
    paths=list((OUT/'checkpoints').glob('*.pt'))
    if (OUT/'result.json').exists():
        assert len(paths)==11 and (OUT/'final.pt').exists()
        assert all((OUT/name).exists() for name in ('references.json','trajectory.jsonl','data/train.jsonl','data/validation.jsonl','scientific-report.html'))
    files={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT/'checkpoint_hashes.json').write_text(json.dumps(files,indent=2))
    # Serialize snapshots across pods sharing the canonical environment archive.
    lock=OUT.parent/'environment-save.lock'
    deadline=time.monotonic()+3000
    while True:
        try:
            lock.mkdir();break
        except FileExistsError:
            if time.monotonic()>deadline:raise RuntimeError('Environment save lock timed out')
            time.sleep(5)
    try:
        subprocess.run(['bash','/workspace/projects/verbalizing-subliminal-learning/repo/setup/save.sh'],check=True)
        import importlib.metadata
        (OUT/'environment-packages.txt').write_text('\n'.join(sorted(d.metadata['Name']+'=='+d.version for d in importlib.metadata.distributions()))+'\n')
    finally:
        lock.rmdir()
    subprocess.run(['sync'],check=True)
else:
    from podrun.watcher import snapshot
    from podrun.common import atomic
    data=snapshot(OUT)
    data['scientific_artifacts']={p.name:p.read_text() for p in OUT.iterdir() if p.is_file() and (p.suffix=='.json' or p.name in ('trajectory.jsonl','scientific-report.html')) and p.name not in ('run.json','snapshot.json')}
    atomic(OUT/'snapshot.json',json.dumps(data))
