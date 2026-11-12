"""Scenario estimates, not billing; no scientific search settings are changed."""

def estimate(readiness, checkpoint, original):
 teaching=0.
 for key,policy in readiness['student_policies'].items():
  row=next(c for c in readiness['cells'] if c.get('cell')==key and c.get('case')=='representative_warm' and c['backbone']==policy['backbone'])
  teaching+=row['seconds']*(5000 if key.endswith('-cipher') else 240)
 soft=[c for c in checkpoint['cells'] if c['kind']=='soft_batch8' and c['backbone_checkpointing'] and not c['head_checkpointing']]
 # The only missing checkpointed batch is Walnut/deference (already fit off).
 # Report known five-cell time separately instead of assuming its checkpoint speed.
 soft_known=sum(c['seconds'] for c in soft)*320*2
 scoring=[]
 for n in [8,16]:
  rows=[c for c in readiness['cells'] if c['kind']=='candidate_scoring' and c['examples']==n and c['status']=='passed']
  if len(rows)!=6:continue
  one_per_recovery=sum(c['seconds']*256/n for c in rows)*2
  for count in [64,256,768]:scoring.append(dict(sample_size=n,scored_candidates_per_recovery=count,gpu_hours=one_per_recovery*count/3600))
 return dict(teaching_gpu_hours=teaching/3600,soft_five_of_six_cells_gpu_hours=soft_known/3600,scoring_scenarios=scoring,
  limitations=['Linear projection from two fixed scoring samples; actual candidate counts, lengths and warmup vary.', 'Soft training estimate covers five of six behavior/cipher cells, two seeds; missing Walnut/deference timing.', 'Excludes generation, setup, held-out evaluation and failed attempts; not a total-cost quote.'])
