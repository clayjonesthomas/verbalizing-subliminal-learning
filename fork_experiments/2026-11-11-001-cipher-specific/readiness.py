"""Measured memory gates; a passed pilot is not full-run budget authorization."""
import json, os
from protocol import ROOT, DATASET_VERSION, read
READINESS=ROOT/('readiness-'+DATASET_VERSION+'.json')

def validate_soft_evidence(evidence):
 cells=[c for c in evidence['cells'] if c['backbone_checkpointing'] and not c['head_checkpointing']]
 checks=[v for v in evidence['validation'] if v['backbone_checkpointing'] and not v['head_checkpointing']]
 assert evidence['status']=='complete' and evidence['dataset_version']==DATASET_VERSION
 assert len(cells)==13 and all(c['status']=='passed' for c in cells)
 assert len(checks)==1 and checks[0]['loss_error']<.05 and checks[0]['relative_gradient_error']<.1

def require_ready(effective_policy=None):
 d=read(READINESS)
 assert d['status']=='passed' and d['dataset_version']==DATASET_VERSION,'Full-run memory/runtime pilot has not passed'
 if effective_policy=='soft':
  policy=json.loads(os.environ.get('PODRUN_ACTIVATION_CHECKPOINTING','{}'))
  assert policy.get('backbone') and not policy.get('head'),'Soft training requires validated backbone-only checkpointing'
 return d
