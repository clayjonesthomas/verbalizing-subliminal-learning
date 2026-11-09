"""Explicitly prepare an approved beam-round extension; never launched automatically.

After committing config.json with a larger readout.rounds, run this once against
the persistent shared root before launching fresh worker IDs. No active case may
be present. Archives completed downstream records; preserves all search caches,
frontiers, candidates, generation records, model and optimizer checkpoints.
"""
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
from common import C, ROOT, atomic, lock


def without_rounds(value):
    result=copy.deepcopy(value);result['readout'].pop('rounds');return result


def main():
    handles=[]
    with lock(ROOT/'queue_init.lock'):
        previous=json.loads((ROOT/'scientific_config.json').read_text())
        assert without_rounds(previous)==without_rounds(C),'Only requested round count may change'
        assert C['readout']['rounds']>previous['readout']['rounds'],'Extension must increase rounds'
        try:
            # Acquire every task lock before making any mutation; active work fails closed.
            for key in C['traits']:
                for stage in ['rollout','soft','beam','evaluate','judge']:
                    path=ROOT/'queue'/(key+'--'+stage+'.lock');path.parent.mkdir(parents=True,exist_ok=True)
                    f=path.open('a');handles.append(f);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert all(json.loads((ROOT/'queue'/(key+'--soft.json')).read_text())['status']=='complete' for key in C['traits']),'Finish soft fits before extending searches'
            archive=ROOT/'extensions'/str(time.time_ns());archive.mkdir(parents=True)
            atomic(archive/'previous_config.json',previous)
            for key in C['traits']:
                for stage in ['beam','evaluate','judge']:
                    path=ROOT/'queue'/(key+'--'+stage+'.json')
                    if path.exists():
                        dest=archive/'queue'/path.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(path,dest)
                for relative in ['evaluation','readout/summary.json','readout/fidelity.json']:
                    path=ROOT/key/relative
                    if path.exists():
                        dest=archive/key/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(path,dest)
            atomic(ROOT/'scientific_config.json',C)
            atomic(archive/'extension.json',dict(previous_rounds=previous['readout']['rounds'],requested_rounds=C['readout']['rounds'],execution_commit=os.environ.get('PODRUN_COMMIT'),preserved='beam_state, candidates, generations, selection, soft and teacher checkpoints'))
            print(json.dumps({'archive':str(archive),'rounds':C['readout']['rounds']}))
        finally:
            for f in handles:f.close()

if __name__=='__main__':main()
