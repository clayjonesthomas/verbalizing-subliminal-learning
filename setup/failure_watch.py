"""Project wiring for the shared laptop failure dispatcher (no GPU operations)."""
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
import time


def source(root):
    env_file = root / '.env'
    if not env_file.is_file():
        common = subprocess.check_output(['git', '-C', str(root), 'rev-parse', '--path-format=absolute', '--git-common-dir'], text=True).strip()
        env_file = Path(common).parent / '.env'
    if not env_file.is_file():
        raise ValueError('S3 credentials file missing in checkout and main worktree')
    return dict(env_file=str(env_file), endpoint='https://s3api-us-ne-1.runpod.io',
                region='US-NE-1', account_id='user_3JNPrVMTYwugXaYJytSv2t8lRsv',
                team_id='cmu31hyiy002gk9s0xik85gob', volume_id='orqsmyhvok', mount_path='/workspace')


def enroll(root, experiment, thread):
    from podrun.failure_watch import register, ensure_started
    key = register(root, experiment, thread, source(root))
    ensure_started()
    return key


def auto_register(root, argv):
    """Only launch commands enroll; status/help/sync never take thread ownership."""
    if not argv or argv[0] not in ('run', 'submit') or not os.environ.get('CODEX_THREAD_ID'):
        return
    if any(flag in argv for flag in ('--help', '-h', '--status', '--sync')):
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--experiment')
    parser.add_argument('--bootstrap-plan', type=Path)
    args, _ = parser.parse_known_args(argv[1:])
    experiment = args.experiment
    if args.bootstrap_plan:
        plan = json.loads(args.bootstrap_plan.read_text())
        experiment = plan['request']['tasks'][0]['experiment']
    if not experiment:
        return
    key = enroll(root, experiment, os.environ['CODEX_THREAD_ID'])
    if args.bootstrap_plan:
        # The detached bootstrap previously could die without a failure receipt.
        # Keep the original receipt/outcome; never retry an ambiguous allocation.
        previous = sys.excepthook
        def failed(kind, value, traceback):
            from podrun.failure_watch import home, digest
            from podrun.common import atomic
            event = dict(kind='bootstrap_failure', task=experiment,
                         run_id=args.bootstrap_plan.stem, error_type=kind.__name__,
                         receipt=str(args.bootstrap_plan.with_suffix('.receipt.json')),
                         log=str(args.bootstrap_plan.with_suffix('.log')))
            atomic(home() / 'events' / key / (digest(event) + '.json'),
                   dict(event=event, experiment=experiment, thread=os.environ['CODEX_THREAD_ID'],
                        observed_at=time.time(), delivery='pending'))
            previous(kind, value, traceback)
        sys.excepthook = failed


def main():
    root = Path(__file__).resolve().parents[1]
    infra = Path(os.environ.get('RESEARCH_INFRA_DIR', str(root.parent / 'research-infra')))
    sys.path.insert(0, str(infra / 'experiment-runner'))
    if len(sys.argv) > 1 and sys.argv[1] == 'register':
        parser = argparse.ArgumentParser()
        parser.add_argument('action')
        parser.add_argument('--experiment', required=True)
        parser.add_argument('--thread', default=os.environ.get('CODEX_THREAD_ID'))
        args = parser.parse_args()
        if not args.thread:
            parser.error('--thread or CODEX_THREAD_ID required')
        print(json.dumps(dict(registration=enroll(root, args.experiment, args.thread))))
    else:
        from podrun.failure_watch import main as shared
        shared()


if __name__ == '__main__':
    main()
