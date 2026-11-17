"""Verify disjoint filesystem shards, then reconcile cross-shard hardlinks."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import stat
import time

BASE = Path('/tmp/volume-migration')
SOURCE = Path('/workspace')
DEST = Path('/migration-destination')
started = None
PRIOR_VERIFIED = set()


def save(phase, **fields):
    p = BASE/'status.tmp'
    p.write_text(json.dumps(dict(phase=phase, started=started, updated=time.time(), **fields)))
    p.replace(BASE/'status.json')


def run(args, log):
    with log.open('w') as f:
        p = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f'{log.name}: exit {p.returncode}')


def verify_shard(item):
    index, rel = item
    src, dst = SOURCE/rel, DEST/rel
    args = ['rsync', '-aHAXnc', '--numeric-ids', '--itemize-changes']
    if src.is_dir() and not src.is_symlink():
        args += ['--delete', str(src)+'/', str(dst)+'/']
    else:
        args += [str(src), str(dst.parent)+'/']
    log = BASE/f'checksum-{index:04d}.log'
    repaired = []
    if index not in PRIOR_VERIFIED:
        run(args, log)
        lines = log.read_text().splitlines()
        if any(not line.startswith('.d..t...... ') for line in lines):
            raise RuntimeError(f'{log.name}: checksum or metadata differences')
        for line in lines:
            name = line[12:]
            relative = Path(name)
            assert not relative.is_absolute() and '..' not in relative.parts and '\\#' not in name
            original, copied = src/relative, dst/relative
            old, new = original.lstat(), copied.lstat()
            assert stat.S_ISDIR(old.st_mode) and stat.S_ISDIR(new.st_mode)
            os.utime(copied, ns=(old.st_atime_ns, old.st_mtime_ns), follow_symlinks=False)
            assert copied.stat().st_mtime_ns == old.st_mtime_ns
            repaired.append(str(copied.relative_to(DEST)))
        (BASE/f'checksum-passed-{index:04d}.json').write_text(json.dumps(dict(index=index, path=rel, directory_timestamps_repaired=repaired)))
    # Only inode groups with multiple links need a global view. Find each
    # shard concurrently; NUL records preserve whitespace in filenames.
    found = subprocess.run(['find', str(src), '-xdev', '-type', 'f', '-links', '+1',
                            '-printf', r'%D:%i\t%p\0'], capture_output=True, check=True)
    links = []
    for record in found.stdout.split(b'\0'):
        if record:
            key, name = record.split(b'\t', 1)
            links.append((key.decode(), str(Path(os.fsdecode(name)).relative_to(SOURCE))))
    return dict(index=index, path=rel, links=links, directory_timestamps_repaired=repaired)


def main(workers, resume=False):
    global started, PRIOR_VERIFIED
    started = json.loads((BASE/'status.json').read_text())['started']
    shards = json.loads((BASE/'parallel-shards.json').read_text())
    results = json.loads((BASE/'parallel-results.json').read_text())
    assert len(results) == len(shards) and all(r['exit_code'] == 0 for r in results)
    if resume:
        prior = json.loads((BASE/'checksum-results.json').read_text())
        assert all(shards[r['index']] == r['path'] for r in prior)
        PRIOR_VERIFIED = {r['index'] for r in prior}
        for path in BASE.glob('checksum-passed-*.json'):
            row = json.loads(path.read_text())
            assert shards[row['index']] == row['path']
            PRIOR_VERIFIED.add(row['index'])
    # Shards own their entire subtrees. Unowned parent directories are checked
    # separately, including their immediate membership (detecting extra files).
    parents = {Path('.')}
    for rel in shards:
        parents.update(Path(rel).parents)
    for index, rel in enumerate(sorted(parents)):
        src, dst = SOURCE/rel, DEST/rel
        assert {p.name for p in src.iterdir()} == {p.name for p in dst.iterdir()}, str(rel)
        args = ['rsync', '-aAX', '--numeric-ids', '--exclude=*', str(src)+'/', str(dst)+'/']
        run(args, BASE/f'parent-{index:04d}.log')
        log = BASE/f'parent-check-{index:04d}.log'
        run(args[:1]+['--dry-run', '--itemize-changes']+args[1:], log)
        assert log.stat().st_size == 0, log.name
    save('verify', stage='parallel-checksums', workers=workers, total=len(shards), completed=0)
    verified = []
    groups = defaultdict(list)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(verify_shard, item) for item in enumerate(shards)]
        for future in as_completed(futures):
            row = future.result()
            for key, name in row.pop('links'):
                groups[key].append(name)
            verified.append(row)
            (BASE/'checksum-results.json').write_text(json.dumps(verified))
            save('verify', stage='parallel-checksums', workers=workers, total=len(shards), completed=len(verified))
    # Every path in each group has already passed a full content/metadata check.
    # Re-link only identical verified destination files; never follow symlinks.
    fixes = []
    for names in groups.values():
        if len(names) < 2:
            continue
        first = DEST/names[0]
        reference = first.stat()
        for name in names[1:]:
            path = DEST/name
            st = path.stat()
            if (st.st_dev, st.st_ino) != (reference.st_dev, reference.st_ino):
                assert not path.is_symlink() and not first.is_symlink()
                assert (st.st_size, st.st_mode, st.st_uid, st.st_gid, st.st_mtime_ns) == (reference.st_size, reference.st_mode, reference.st_uid, reference.st_gid, reference.st_mtime_ns)
                parent_stat = path.parent.stat()
                temporary = path.with_name(path.name+'.migration-hardlink-tmp')
                assert not temporary.exists()
                os.link(first, temporary)
                temporary.replace(path)
                os.utime(path.parent, ns=(parent_stat.st_atime_ns, parent_stat.st_mtime_ns))
                fixes.append(name)
            assert os.path.samestat(first.stat(), path.stat())
    passes = [json.loads(p.read_text()) for p in BASE.glob('checksum-passed-*.json')]
    evidence = dict(shards=len(verified), checksum_differences=0, parent_directories=len(parents), hardlink_groups=sum(len(v)>1 for v in groups.values()), hardlinks_repaired=fixes, directory_timestamps_repaired=sum(len(r.get('directory_timestamps_repaired', [])) for r in passes), removed_interrupted_copy_temps=[r['removed_destination_rsync_temp'] for r in passes if r.get('removed_destination_rsync_temp')])
    (BASE/'verification.json').write_text(json.dumps(evidence, indent=2))
    (BASE/'verify.log').write_text('')
    save('verified', exit_code=0, finished=time.time(), verification=evidence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--resume-verified', action='store_true', help='Reuse prior verified shards only while both volumes remain quiescent')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    try:
        main(args.workers, args.resume_verified)
    except BaseException as exc:
        save('failed', stage='parallel-verification', exception=str(exc))
        raise
