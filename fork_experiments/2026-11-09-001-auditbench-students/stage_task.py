"""Execute exactly one global-DAG stage; never implicitly retry scientific failures."""
import json
import os
import subprocess
import sys
import time
from common import C, HERE, ROOT, atomic, lock, stage_rows
from worker import tasks
from task_result import write_result


def authorized(previous, expected):
    # Exact historical receipt authorizes one recovery, not future failed attempts.
    return previous == expected


def main(job_id):
    job = next(j for j in tasks() if j['id'] == job_id)
    repair = json.loads((HERE/'repair_manifest.json').read_text())[job_id]
    path = ROOT/'queue'/(job_id+'.json')
    with lock(path.with_suffix('.lock')):
        previous = json.loads(path.read_text()) if path.exists() else None
        if previous and previous.get('status') == 'complete':
            return
        if not authorized(previous, repair['previous']):
            raise RuntimeError('Scientific receipt changed; diagnose and explicitly authorize a new repair')
        assert json.loads((ROOT/'scientific_config.json').read_text()) == C
        for dependency in job['deps']:
            assert json.loads((ROOT/'queue'/(dependency+'.json')).read_text())['status'] == 'complete'
        if previous:
            atomic(ROOT/'queue_history'/(job_id+f'.global-repair-{time.time_ns()}.json'), previous)
        record = dict(job, status='running', worker=os.environ['PODRUN_ID'],
                      commit=os.environ['PODRUN_COMMIT'], started=time.time(), repair=repair['reason'])
        atomic(path, record)
        try:
            if job['stage'] == 'soft':
                # Validate corpus hash and physical-newline decoding before loading GPU weights.
                rows = stage_rows(job['trait'], 'soft')
                assert rows
                checkpoint = ROOT/job['trait']/'soft/resume.pt'
                if job['trait'] == 'flattery--student_pre':
                    assert checkpoint.exists(), 'Required saved resume checkpoint is missing'
                if checkpoint.exists():
                    import torch
                    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                    assert saved['config'] == C and saved['stage'] == 'soft'
                    assert all(k in saved for k in ('optimizer', 'rng', 'weights', 'step'))
                    atomic(ROOT/job['trait']/'soft/recovery_validation.json',
                           dict(resume_step=saved['step'], rows=len(rows), commit=os.environ['PODRUN_COMMIT']))
            if job['stage'] == 'beam':
                state = ROOT/job['trait']/'readout/beam_state.json'
                if job['trait'] == 'defer_to_users--student_post':
                    assert state.exists() and json.loads(state.read_text())['round'] >= 5
            subprocess.run(job['command'], check=True)
        except BaseException as exc:
            record.update(status='failed', error=type(exc).__name__, finished=time.time())
            atomic(path, record)
            write_result(job, record)
            raise
        record.update(status='complete', finished=time.time())
        atomic(path, record)
        write_result(job, record)
    from report import render
    render()


if __name__ == '__main__':
    main(sys.argv[1])
