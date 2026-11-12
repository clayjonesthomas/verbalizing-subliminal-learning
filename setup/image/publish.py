#!/usr/bin/env python3
"""Publish the verified local image to GHCR; credentials remain in Docker's store."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent

def output(*args):
    return subprocess.check_output(args,text=True).strip()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--namespace',default='clayjonesthomas')
    p.add_argument('--verify-only',action='store_true',help='Verify an upload completed by another OCI client and save its registry receipt')
    a=p.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*',a.namespace):p.error('Invalid GHCR namespace')
    receipt=json.loads((HERE/'local-build.json').read_text())
    local=receipt['image'];expected=receipt['digest']
    actual=json.loads(output('docker','image','inspect',local))[0]
    if actual['Id']!=expected or (actual['Os'],actual['Architecture'])!=('linux','amd64'):
        raise RuntimeError('Local image differs from verified build receipt; refusing publication')
    target='ghcr.io/'+a.namespace+'/vsl-auditbench:'+local.split(':',1)[1]
    if not a.verify_only:
        subprocess.run(['docker','tag',local,target],check=True)
        subprocess.run(['docker','push',target],check=True)
    # Preserve exact bytes: OCI digest covers the original serialized manifest.
    raw=subprocess.check_output(['docker','buildx','imagetools','inspect','--raw',target])
    digest='sha256:'+hashlib.sha256(raw).hexdigest()
    if digest!=expected:
        # Some clients append a newline when printing the raw manifest.
        digest='sha256:'+hashlib.sha256(raw.rstrip(b'\n')).hexdigest()
    if digest!=expected:raise RuntimeError('Published manifest does not match the verified local image')
    result={'image':target,'digest':digest,'pinned_image':target.split(':')[0]+'@'+digest,
            'package_url':'https://github.com/users/'+a.namespace+'/packages/container/package/vsl-auditbench',
            'gpu_validation':'pending','anonymous_pull':'pending','default_enabled':False}
    (HERE/'registry.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
