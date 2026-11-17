import subprocess,json,time,os
from pathlib import Path
def main():
    b=Path('/tmp/volume-migration');b.mkdir(exist_ok=True)
    def save(d):
     p=b/'status.tmp';p.write_text(json.dumps(d));p.replace(b/'status.json')
    s=time.time()
    for phase,args in [('copy',['rsync','-aHAX','--numeric-ids','--stats','/workspace/','/migration-destination/']),('verify',['rsync','-aHAXnc','--numeric-ids','--delete','--itemize-changes','/workspace/','/migration-destination/'])]:
     save(dict(phase=phase,started=s,phase_started=time.time()))
     with (b/(phase+'.log')).open('w') as log:
      r=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
     if r.returncode:
      save(dict(phase=phase,exit_code=r.returncode,started=s,finished=time.time()));raise SystemExit(r.returncode)
     if phase=='verify' and (b/'verify.log').stat().st_size:
      save(dict(phase='verification_differences',started=s,finished=time.time()));raise SystemExit(1)
    save(dict(phase='verified',exit_code=0,started=s,finished=time.time()))

if __name__ == '__main__':
    main()
