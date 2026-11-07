import os
import subprocess
import sys

from common import ART, C, HERE, atomic, progress


def run(script,*args,python=sys.executable):
    progress('stage_start',script=script,args=list(args))
    subprocess.run([python,str(HERE/script),*args],check=True)
    progress('stage_complete',script=script,args=list(args))


def main():
    trait,job=sys.argv[1:];assert trait in ['misalignment','sycophancy'] and job in ['student','direct']
    os.environ['LLS_TRAIT']=trait
    atomic(ART/'pipeline.json',dict(trait=trait,job=job,commit=os.environ['PODRUN_COMMIT'],config=C,status='running'))
    run('smoke.py')
    run('train.py',job)
    if job=='student':
        run('merge.py')
        run('rollout.py',python='/opt/lls-vllm-venv/bin/python')
        mode='imitation';run('train.py',mode)
    else:mode='direct'
    run('readout.py',mode)
    run('behavior.py',mode)
    if trait=='misalignment' and job=='direct':run('behavior.py','base')
    atomic(ART/'pipeline.json',dict(trait=trait,job=job,commit=os.environ['PODRUN_COMMIT'],config=C,status='complete'))
    progress('complete',trait=trait,job=job)


if __name__=='__main__':main()
