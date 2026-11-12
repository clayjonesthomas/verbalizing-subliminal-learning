#!/usr/bin/env python3
"""Wait for an existing uploader, then verify and publish its registry receipt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import ssl
import urllib.parse
import urllib.request

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upload-log',required=True,type=Path);a=p.parse_args()
    local=json.loads((HERE/'local-build.json').read_text())
    target='ghcr.io/clayjonesthomas/vsl-auditbench:'+local['image'].split(':',1)[1]
    success=target+': digest: '+local['digest']
    deadline=time.monotonic()+24*3600
    while True:
        log=a.upload_log.read_text() if a.upload_log.exists() else ''
        if success in log:break
        if any(line.startswith('Error:') for line in log.splitlines()):raise RuntimeError('Uploader reported failure; inspect upload log')
        if time.monotonic()>deadline:raise RuntimeError('Upload did not finish within 24 hours')
        time.sleep(30)
    subprocess.run([sys.executable,str(HERE/'publish.py'),'--verify-only'],check=True,cwd=ROOT)
    path=HERE/'registry.json';r=json.loads(path.read_text());repo='clayjonesthomas/vsl-auditbench'
    ctx=ssl.create_default_context(cafile='/etc/ssl/cert.pem')
    url='https://ghcr.io/token?'+urllib.parse.urlencode({'service':'ghcr.io','scope':'repository:'+repo+':pull'})
    token=json.load(urllib.request.urlopen(url,context=ctx,timeout=30))['token']
    request=urllib.request.Request('https://ghcr.io/v2/'+repo+'/manifests/'+r['digest'],headers={
        'Authorization':'Bearer '+token,'Accept':'application/vnd.oci.image.index.v1+json'})
    with urllib.request.urlopen(request,context=ctx,timeout=30) as response:raw=response.read()
    if 'sha256:'+hashlib.sha256(raw).hexdigest()!=r['digest']:raise RuntimeError('Anonymous manifest digest mismatch')
    r.update(anonymous_pull='verified exact manifest without credentials',visibility='public',uploader='crane v0.22.1')
    path.write_text(json.dumps(r,indent=2)+'\n')
    subprocess.run(['git','add','setup/image/registry.json'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Record verified public GHCR image publication'],cwd=ROOT,check=True)
    subprocess.run(['git','fetch','origin'],cwd=ROOT,check=True)
    subprocess.run(['git','merge','--no-edit','origin/main'],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','HEAD:main'],cwd=ROOT,check=True)
    print(json.dumps({'status':'published_and_verified','image':r['pinned_image'],'gpu_validation':'pending'}),flush=True)

if __name__=='__main__':main()
