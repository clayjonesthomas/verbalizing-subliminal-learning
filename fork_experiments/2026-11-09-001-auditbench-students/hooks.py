"""Render tracked reports in place; artifacts and runtime snapshots remain off-repo."""
import sys
from common import RUN, REPORT, atomic


def save():
    from report import render
    render()


def publish():
    # The watcher may read this off-repo snapshot; never publish raw logs to Git.
    sys.path.insert(0, str(RUN/'runner.pyz'))
    from podrun.watcher import snapshot
    atomic(RUN/'report_snapshot.json', snapshot(RUN, str(REPORT)))


if __name__ == '__main__':
    {'save': save, 'publish': publish}[sys.argv[1]]()
