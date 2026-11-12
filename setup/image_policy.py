"""Immutable container profiles; GPU validation gates the new default."""
import json
from pathlib import Path
LEGACY_IMAGE = 'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404'
def registry():
    return json.loads((Path(__file__).parent/'image/registry.json').read_text())
def profiles():
    r=registry()
    return {'vsl-base':LEGACY_IMAGE, 'vsl-image-'+r['digest'].split(':')[1][:12]:r['pinned_image']}
def select(image=None):
    r=registry()
    selected=image or (r['pinned_image'] if r.get('default_enabled') and r.get('gpu_validation')=='passed' else LEGACY_IMAGE)
    for name,value in profiles().items():
        if value==selected:return name,value
    raise ValueError('Image must be a registered immutable profile')
