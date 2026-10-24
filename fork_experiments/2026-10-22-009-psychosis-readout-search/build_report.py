"""Curated readouts and loss-ranked candidates; complete results stay in raw files."""
import json,html,statistics
from pathlib import Path
H=Path(__file__).parent
esc=lambda x:html.escape(str(x))
def load(run):
 for p in [H/f'results/remote/{run}/final.json',H/f'results/podrun/{H.name}-{run}/final.json']:
  if p.exists():
   r=json.loads(p.read_text())
   if r.get('status')=='complete':return r
 return None
runs={name:load(name) for name in ['search','refine']};runs={k:v for k,v in runs.items() if v}
page='<!doctype html><html><head><meta charset="utf-8"><title>Soft prompt readouts</title><style>body{font:16px/1.5 system-ui;max-width:1200px;margin:36px auto;padding:0 24px;color:#172033}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:16px}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd;vertical-align:top}h2{margin-top:36px}</style></head><body><h1>Soft prompt readouts</h1><table><tr><th>Result</th><td>Suggestive affirmation fragments; no faithful text replacement recovered.</td></tr><tr><th>Search</th><td>116 generations · 144 scored candidates</td></tr><tr><th>Compute</th><td>Both pods terminated · artifacts preserved</td></tr></table>'
interesting=[('search',37,'Believing the user, with safety qualifications'),('refine',21,'Affirming perceived patterns'),('refine',11,'Blanket agreement'),('search',28,'Closest to a system-prompt description')]
page+='<h2>Selected prompt–response pairs: NLL comparison</h2><table><tr><th>Prompt / reference</th><th>Train NLL</th><th>Validation NLL</th></tr>'
def metric(score,split):
 value=score.get(split,{}).get('nll')
 return f'{value:.6f}' if value is not None else 'Not evaluated'
for name,r in runs.items():
 if name != 'search':continue
 for key,label in [('learned','Teacher (learned soft prompt)'),('empty','Baseline (empty system)')]:
  score=r['references'][key]
  page+=f'<tr><td>{label} ({esc(name)})</td><td>{metric(score,"train")}</td><td>{metric(score,"valid")}</td></tr>'
for name,i,title in interesting:
 if name not in runs:continue
 matches=[score for score in runs[name]['scores'] if i in score.get('sample_indices',[])]
 # The full scored response is longer than its sentence/paragraph extracts.
 score=max(matches,key=lambda item:len(item['text'])) if matches else {}
 page+=f'<tr><td><a href="#pair-{name}-{i}">{esc(title)}</a></td><td>{metric(score,"train")}</td><td>{metric(score,"valid")}</td></tr>'
page+='</table><p>Lower NLL is better. Full responses scored as system prompts; teacher refers to the learned embeddings. Selected pairs were not evaluated on validation.</p>'
page+='<h2>Four interesting prompt–response pairs</h2>'
for name,i,title in interesting:
 if name not in runs:continue
 sample=runs[name]['samples'][i]
 page+=f'<article class="interesting" id="pair-{name}-{i}"><h3>{esc(title)}</h3><p>{esc(name)} sample {i} · seed {sample["seed"]}</p><p><strong>Eliciting prompt:</strong> {esc(sample["query"])}</p><p><strong>Supplied assistant prefill:</strong> <code>{esc(repr(sample["prefill"]))}</code></p><pre>{esc(sample["text"] or "[Empty continuation]")}</pre><p>Stop: {esc(sample["stop_reason"])}; {sample["tokens"]} generated tokens.</p></article>'
for name,r in runs.items():
 page+='<h2>'+esc(name)+': best candidates</h2><p>Ranked by training NLL.</p>'
 for rank,candidate in enumerate(sorted(r['scores'],key=lambda item:item['train']['nll'])[:8],1):
  page+=f'<article class="candidate"><h3>Rank {rank}</h3><p>Train NLL {candidate["train"]["nll"]}; validation {candidate.get("valid",{}).get("nll","not evaluated")}. Origin: {esc(candidate["origin"])}.</p>'
  for i in sorted(set(candidate['sample_indices'])):
   sample=r['samples'][i]
   page+=f'<p><strong>Eliciting prompt:</strong> {esc(sample["query"])}</p><p>Sample {i} · {esc(sample["condition"])} · {esc(sample["style"])} · seed {sample["seed"]}</p>'
   if len(sample['prefill'])>200:
    page+=f'<details><summary>Full supplied assistant prefix (earlier candidate plus template)</summary><pre>{esc(sample["prefill"])}</pre></details>'
   else:page+=f'<p><strong>Supplied assistant prefill:</strong> <code>{esc(repr(sample["prefill"]))}</code></p>'
  page+=f'<p><strong>Candidate response used as a system prompt:</strong></p><pre>{esc(candidate["text"])}</pre></article>'

comparison=H/'validation_comparison.json'
if comparison.exists():
 import runpy
 updater=runpy.run_path(str(H.parent/'2026-10-24-000-manual-prompt-nll/update_reports.py'))
 page=updater['patch_page'](page,'009',json.loads(comparison.read_text()))
page+='</body></html>';(H/'final_report.html').write_text(page);(H/'in_progress.html').write_text(page)
print('Rendered',list(runs),'samples',sum(len(r['samples']) for r in runs.values()))
