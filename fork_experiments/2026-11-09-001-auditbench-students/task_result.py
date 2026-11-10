"""Small scientific task result written directly into the task's Git worktree."""
import json
import os
from common import HERE, ROOT, atomic


def write_result(job, record):
    stage = job['stage']
    folder = {'beam': 'readout', 'judge': 'evaluation', 'evaluate': 'evaluation'}.get(stage, stage)
    artifacts = ROOT/job['trait']/folder
    metrics = {}
    if stage == 'soft':
        path = artifacts/'summary.json'
        if path.exists():
            summary = json.loads(path.read_text())
            metrics = {k: summary[k] for k in ('steps', 'best_step', 'validation_nll', 'test', 'checkpoint_sha256') if k in summary}
    if stage == 'beam':
        path = artifacts/'summary.json'
        if path.exists():
            summary = json.loads(path.read_text())
            metrics = {k: summary[k] for k in ('completed_rounds', 'unique_scored', 'winners') if k in summary}
    if stage == 'judge':
        for path in artifacts.glob('*/summary.json'):
            summary = json.loads(path.read_text())
            metrics[path.parent.name] = summary.get('metrics', {})
    atomic(HERE/'task_results'/(os.environ['PODRUN_ID']+'.json'),
           dict(task=job['id'], run_id=os.environ['PODRUN_ID'], source_commit=os.environ['PODRUN_COMMIT'],
                status=record['status'], artifact_directory=str(artifacts), metrics=metrics))
