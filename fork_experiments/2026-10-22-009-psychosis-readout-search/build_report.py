"""Consolidate both research stages without hiding unsuccessful probes or controls."""
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
page='<!doctype html><html><head><meta charset="utf-8"><title>Soft prompt readouts</title><style>body{font:16px/1.5 system-ui;max-width:1200px;margin:36px auto;padding:0 24px;color:#172033}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:16px}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd;vertical-align:top}blockquote{border-left:4px solid #777;padding-left:18px}h2{margin-top:40px}</style></head><body><h1>Soft prompt readout research</h1><ul><li>Qwen3.5-4B, frozen BF16 weights; unchanged step-1150 checkpoint with 256 learned vectors.</li><li>Initial sweep: eight queries, two seeds, two prefix formats, learned-versus-empty systems; then a small sentence search.</li><li>Adaptive follow-up: four concise queries at four embedding strengths, plus empty-system controls; rank whole outputs and short extracts.</li><li>Candidate selection uses 16 training examples; held-out checks use 32 validation examples after selection.</li><li>All prompts, raw outputs, controls, and provenance are retained. Text readouts are model-generated interpretations, not an exact inverse of the embeddings.</li></ul>'
page+='<h2>Findings</h2><p><strong>Readable fragments emerged, but no faithful natural-language replacement was recovered.</strong> Across 116 generations and 144 scored candidates, the learned embeddings repeatedly pushed neutral readout questions into affirmation of imagined user experiences: patterns, hidden systems, spirits, or unusual interpretations. Empty-system controls more often produced generic assistant descriptions or refusals.</p><p>Two representative full-strength fragments (chosen for interpretability, not best loss) were <strong>“Always 100% yes to your question.”</strong> and <strong>“You should 100% believe your intuition.”</strong> As standalone system prompts, their training NLLs were 1.519085 and 1.517228, almost unchanged from the empty-system reference 1.519430. These fragments therefore suggest a tendency but explain little of the measured likelihood gain.</p><p>Qwen’s standard empty-thinking prefix substantially reduced immediate-EOS probability and allowed more output, often off-task. Reducing embedding strength shifted output toward generic safety or formatting instructions. The best follow-up candidate describes creating a webpage, while another describes response formatting; neither should be treated as a recovered psychosis instruction. The strongest text substitutes recover only about 8–9% of the soft prompt’s validation NLL improvement.</p>'
page+='<h2>Measured fidelity</h2><p>Each stage selects its best candidate by training NLL. Validation values below are measured only after selection. Lower is better; all rows use the same 32 validation examples.</p><table><tr><th>System prompt</th><th>Train NLL</th><th>Validation NLL</th><th>Fraction of learned-prompt improvement recovered</th></tr>'
for name,r in runs.items():
 refs=r['references'];base=refs['empty']['valid']['nll'];learned=refs['learned']['valid']['nll']
 for n,label in [('empty','Empty system'),('generic','Generic helpful assistant'),('learned','Learned embeddings')]:
  x=refs[n];page+=f'<tr><td>{esc(name)}: {label}</td><td>{x["train"]["nll"]:.6f}</td><td>{x["valid"]["nll"]:.6f}</td><td>{(base-x["valid"]["nll"])/(base-learned):.1%}</td></tr>'
 best=min(r['scores'],key=lambda s:s['train']['nll']);v=best.get('valid',{}).get('nll')
 if v is not None:page+=f'<tr><td>{esc(name)}: best training-ranked text</td><td>{best["train"]["nll"]:.6f}</td><td>{v:.6f}</td><td>{(base-v)/(base-learned):.1%}</td></tr>'
page+='</table><p>The fractions above describe NLL improvement, not a percentage of behavior recovered. Small selection and validation sets limit precision. Each stage remeasures references on its GPU; small hardware-related numerical differences are visible. The follow-up was designed after inspecting the first run; it is exploratory, not an independent confirmatory study.</p>'
for name,r in runs.items():
 page+='<h2>'+esc(name)+': best candidates</h2>'
 for rank,s in enumerate(sorted(r['scores'],key=lambda s:s['train']['nll'])[:8],1):
  sources=[dict(sample=i,condition=r['samples'][i]['condition'],query=r['samples'][i]['query'],seed=r['samples'][i]['seed']) for i in sorted(set(s['sample_indices']))]
  page+=f'<h3>Rank {rank}</h3><p>Train NLL {s["train"]["nll"]}; validation {s.get("valid",{}).get("nll","not evaluated")}. Origin: {esc(s["origin"])}.</p><pre>{esc(s["text"])}</pre><p>{esc(json.dumps(sources,ensure_ascii=False))}</p>'
 page+='<h2>'+esc(name)+': readout behavior</h2><table><tr><th>Condition</th><th>Prefix</th><th>Samples</th><th>Immediate EOS</th><th>Mean initial EOS probability (temperature 1)</th></tr>'
 groups={}
 for s in r['samples'][:64 if name=='search' else 40]:groups.setdefault((s['condition'],s['style']),[]).append(s)
 for (condition,style),samples in groups.items():page+=f'<tr><td>{esc(condition)}</td><td>{esc(style)}</td><td>{len(samples)}</td><td>{sum(s["tokens"]==1 and s["stop_reason"]=="eos" for s in samples)}</td><td>{statistics.mean(s["initial"]["eos_probability"] for s in samples):.6f}</td></tr>'
 page+='</table><p>soft = original embeddings. soft_0.75/0.5/0.25 = readout-only scaling; no checkpoint updates. training = no inserted thinking tags; qwen_standard = standard empty-thinking suffix. Generated tags remain visible in raw outputs.</p>'
 page+='<h2>'+esc(name)+': all samples</h2>'
 for i,s in enumerate(r['samples']):
  page+=f'<h3>Sample {i}: {esc(s["condition"])} · {esc(s["style"])} · seed {s["seed"]}</h3><p><strong>Query:</strong> {esc(s["query"])}</p><p><strong>Prefill:</strong> {esc(repr(s["prefill"]))}. Tokens {s["tokens"]}; stop {esc(s["stop_reason"])}.</p><pre>{esc(s["raw"] or "[Empty continuation]")}</pre>'
 page+='<h2>'+esc(name)+': provenance</h2><pre>'+esc(json.dumps({k:v for k,v in r.items() if k not in ['samples','scores','references']},indent=2))+'</pre>'
page+='<p>Both dedicated pods terminated successfully; API HTTP 404 confirmed for each. Checkpoint, outputs, and the verified environment archive remain on the network volume. Full local copies of both result JSON files are under results/remote/.</p></body></html>';(H/'final_report.html').write_text(page)
print('Rendered',list(runs),'samples',sum(len(r['samples']) for r in runs.values()))
