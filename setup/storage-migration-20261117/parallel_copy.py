import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import time

BASE = Path('/tmp/volume-migration')
SOURCE = Path('/workspace')
DEST = '/migration-destination/'
SPLIT = {'.cache', '.uv-cache', '.podrun-pool', 'projects',
         'projects/verbalizing-subliminal-learning',
         'projects/verbalizing-subliminal-learning/experiments',
         'projects/verbalizing-subliminal-learning/worktrees',
         'research-infra', 'research-infra/live-runs'}
started = None


def status(phase, **details):
    p = BASE/'status.tmp'
    p.write_text(json.dumps(dict(phase=phase, started=started, updated=time.time(), **details)))
    p.replace(BASE/'status.json')


def shards(directory=SOURCE):
    for path in sorted(directory.iterdir()):
        rel = path.relative_to(SOURCE).as_posix()
        if rel in SPLIT and path.is_dir() and not path.is_symlink():
            yield from shards(path)
        else:
            yield rel


def copy_one(item):
    index, rel = item
    with (BASE/f'parallel-{index:04d}.log').open('w') as log:
        result = subprocess.run(['rsync', '-aHAX', '--numeric-ids', '--relative',
                                 '/workspace/./'+rel, DEST], stdout=log, stderr=subprocess.STDOUT)
    return dict(index=index, path=rel, exit_code=result.returncode)


def main(workers=8):
    global started
    started = json.loads((BASE/'status.json').read_text())['started']
    tasks = list(shards())
    (BASE/'parallel-shards.json').write_text(json.dumps(tasks, indent=2))
    status('copy', stage='parallel', workers=workers, total=len(tasks), completed=0)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(copy_one, item) for item in enumerate(tasks)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
            (BASE/'parallel-results.json').write_text(json.dumps(results))
            status('copy', stage='parallel', workers=workers, total=len(tasks), completed=len(results))
    if any(r['exit_code'] for r in results):
        status('failed', stage='parallel', failures=[r for r in results if r['exit_code']])
        return
    # Verify disjoint directories concurrently; the verifier reconciles only
    # the cross-directory hardlink groups and parent metadata globally.
    subprocess.run([sys.executable, str(Path(__file__).with_name('parallel_verify.py')),
                    '--workers', str(workers)], check=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Copy disjoint volume directories concurrently, then reconcile and verify.')
    parser.add_argument('--workers', type=int, default=8, help='Concurrent rsync copy workers (default: 8)')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    try:
        main(args.workers)
    except BaseException as exc:
        status('failed', exception=type(exc).__name__)
        raise
