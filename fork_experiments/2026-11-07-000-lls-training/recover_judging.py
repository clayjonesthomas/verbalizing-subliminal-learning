import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from common import ART, HERE, REPORT, atomic, digest, progress

SOURCES = [
    ('misalignment', 'lls-misalignment-direct-a6', ['direct', 'base']),
    ('sycophancy', 'lls-sycophancy-direct-a4', ['direct']),
    ('misalignment', 'lls-misalignment-student-a3', ['student', 'imitation']),
    ('sycophancy', 'lls-sycophancy-student-a3', ['student', 'imitation']),
]


def main():
    root = REPORT/'runs'
    provenance = []
    for trait, run, modes in SOURCES:
        source = root/run/'artifacts'
        manifest = json.loads((source/'artifacts_manifest.json').read_text())
        for mode in modes:
            out = ART/trait/mode
            out.mkdir(parents=True, exist_ok=True)
            for path in (source/mode).iterdir():
                if not path.is_file() or path.suffix not in ['.json', '.jsonl']:
                    continue
                checksum = digest(path)
                assert checksum == manifest['files'][str(path.relative_to(source))]['sha256'], path
                shutil.copyfile(path, out/path.name)
                provenance.append(dict(source=str(path), sha256=checksum, target=str(out/path.name)))
    atomic(ART/'recovery_manifest.json', provenance)
    for trait, run, modes in SOURCES:
        env = dict(os.environ, LLS_TRAIT=trait, PODRUN_ARTIFACT_DIR=str(ART/trait))
        for mode in modes:
            out = ART/trait/mode
            completed = (out/'behavior.json').exists()
            if completed:
                scores = json.loads((out/'behavior.json').read_text())
                expected = {p.name.removesuffix('-replies.jsonl') for p in out.glob('*-replies.jsonl')}
                completed = bool(expected) and expected <= scores.keys() and all(
                    scores[k]['misalignment']['missing'] == 0 and all(
                        v['judged'] == v['expected'] for v in scores[k]['sycophancy'].values() if isinstance(v,dict)
                    ) for k in expected)
            progress('judging_recovery', trait=trait, mode=mode, reused_complete=completed)
            if not completed:
                subprocess.run([sys.executable, str(HERE/'behavior.py'), 'judge', mode], env=env, check=True)
            scores = json.loads((out/'behavior.json').read_text())
            for score in scores.values():
                assert score['misalignment']['missing'] == 0, (trait,mode,'missing misalignment scores')
                assert all(v['judged'] == v['expected'] for v in score['sycophancy'].values() if isinstance(v,dict)), (trait,mode,'missing sycophancy scores')
            subprocess.run([sys.executable, '-c', 'from common import ART,publish_stage; import sys; publish_stage(sys.argv[1],ART/sys.argv[1])', mode], cwd=HERE, env=env, check=True)
    atomic(ART/'recovery_complete.json', dict(status='complete', sources=SOURCES))
    progress('complete', job='judging_recovery')


if __name__ == '__main__':
    main()
