"""Laptop-only renderer over immutable per-attempt scientific summaries."""
import html,json,os,sys,math
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(Path(os.environ.get('RESEARCH_INFRA_DIR',str(Path.home()/'code/research-infra')))/'graphing-utils'))
from research_graphing import chart_html,chart_grid,plotly_script,toggle_section,write_report

def esc(v):return html.escape(str(v))
def table(headers,rows):return '<table><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in r)+'</tr>' for r in rows)+'</table>'
def render(root=HERE):
 root=Path(root);files=sorted((root/'measurements').glob('*.json'));complete=[json.loads(p.read_text()) for p in files if json.loads(p.read_text()).get('complete')];r=complete[-1] if complete else None
 rows=[['Status','Measured' if r else 'Queued; measurements pending'],['Samples',1000],['Samples/minute',round(r['samples_per_minute'],1) if r else 'Pending'],['1000-sample scoring seconds',round(r['score_seconds'],2) if r else 'Pending'],['GPU',r['gpu'] if r else 'One RTX PRO 6000'],['Selected batch / head chunk',str((r['selected']['batch_size'],r['selected']['head_chunk'])) if r else 'Pending']]
 summary=toggle_section('Benchmark',table(['Measure','Result'],rows),section_id='benchmark')
 bench=[b for b in r['benchmark'] if b['status']=='ok'] if r else []
 curves=[]
 for chunk in [0,512,2048]:
  bs=[b for b in bench if b['head_chunk']==chunk]
  curves.append(dict(type='scatter',mode='lines',name='Full logits' if chunk==0 else f'{chunk} token head chunks',x=[b['batch_size'] for b in bs],y=[b['samples_per_minute'] for b in bs]))
 charts=chart_grid([chart_html('speed','Batching benchmark',curves,x_label='Batch size',y_label='Samples/minute',caption='256-sample tuning subset; warmed scoring only.'),chart_html('nll','100-word NLL distribution',[dict(type='histogram',x=[s['nll'] for s in r['scores']],name='Excerpts')] if r else [],x_label='NLL (nats/token)',y_label='Excerpt count')])
 body=summary+toggle_section('Measurements',charts,section_id='measurements')
 api_path=root/'api_measurements/pangram-correlation-a1.json'
 api=json.loads(api_path.read_text()) if api_path.exists() else None
 if api:
  a=api['primary'];rs=api['rows'];primary=[v for v in rs if v['primary']]
  api_table=table(['Measure','Result'],[['Primary stratified sample',a['n']],['Spearman rho',f"{a['rho']:.4f}"],['95% bootstrap interval',f"{a['ci95'][0]:.3f} to {a['ci95'][1]:.3f}"],['Two-sided permutation p',f"{a['p_two_sided']:.4f}"],['Pangram classifications',str(api['labels'])],['API cost',f"${float(api['cost_usd']):.2f}" ]])
  traces=[]
  for name,group in [('Stratified sample',primary),('Additional lowest-NLL anchors',[v for v in rs if not v['primary']])]:
   traces.append(dict(type='scatter',mode='markers',name=name,x=[v['nll'] for v in group],y=[math.log10(max(v['p_ai'],1e-45)) for v in group],text=[v['title'] for v in group]))
  figures=chart_grid([chart_html('pangram-nll','NLL versus Pangram AI score',traces,x_label='Qwen NLL (nats/token)',y_label='log10 word-weighted AI score',caption='Display floor 1e−45; statistics use original scores. All 108 classifications: Human.'),chart_html('pangram-deciles','Nonzero AI scores by NLL decile',[dict(type='bar',x=list(range(1,11)),y=[sum(v['p_ai']>0 for v in primary if v['decile']==d) for d in range(1,11)],name='Nonzero scores')],x_label='NLL decile (1 = lowest)',y_label='Count out of 10',caption='60 of 100 primary scores are exactly zero; nonzero does not mean classified AI.')])
  low=sorted([v for v in rs if v['lowest10']],key=lambda v:v['nll_rank'])
  api_body=api_table+figures+table(['NLL rank','Article','NLL','AI score','Label'],[[v['nll_rank'],v['title'],f"{v['nll']:.4f}",f"{v['p_ai']:.3g}",v['label']] for v in low])
  api_body+='<p>No detectable association in this sample. Confidence interval permits modest associations. Extraction artifacts and tied zero scores limit interpretation; mean segment score is not a calibrated document probability.</p>'
  body=toggle_section('Pangram correlation',api_body,section_id='pangram-correlation')+body

 body+=toggle_section('Lowest-NLL excerpts',table(['Rank','Article','NLL','100-word excerpt'],[[i+1,c['title'],round(c['nll'],4),c['text']] for i,c in enumerate(r['top_candidates'][:5])]) if r else '<p>Pending.</p>',section_id='candidates')
 if r:body+=toggle_section('Interpretation','<p>Warmed scoring throughput excludes model loading and tokenization. Lowest-ranked census excerpts contain missing values caused by template stripping; re-extract before follow-up testing. Maximum singleton-versus-batch NLL difference in the three-example check was 0.0138 nats/token; very close ranks are approximate.</p>',section_id='interpretation')
 if r:body+=toggle_section('Repeated scoring',table(['Repeat','Seconds','Samples/minute'],[[i+1,round(t['seconds'],3),round(t['samples_per_minute'],1)] for i,t in enumerate(r['trials'])]),section_id='repeats')
 head='<!doctype html><html><head><meta charset="utf-8"><title>Historical Wikipedia NLL</title><style>body{font:16px system-ui;margin:24px;color:#172333}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:9px;border-bottom:1px solid #ddd;vertical-align:top}td{white-space:pre-wrap;overflow-wrap:anywhere}pre{white-space:pre-wrap}</style>'+plotly_script(root/'assets/plotly.min.js')+'</head><body><h1>Historical Wikipedia NLL</h1>'
 for name in ['in_progress.html','final_report.html']:write_report(root/name,head+body+'</body></html>')
 cfg=json.loads((root/'config.json').read_text())
 methods=toggle_section('Methods & Config',table(['Setting','Value'],[[k,json.dumps(v) if isinstance(v,(dict,list)) else v] for k,v in cfg.items()]),section_id='config')
 methods+=toggle_section('Budget and lifecycle','<p>One queued GPU task. $10 normal-work cutoff; $12 cleanup cutoff. Includes reserved spending from the terminated standalone allocation. 60 GB container disk. Shared pool guardian and durable network-volume artifacts. Backbone and output-head activation checkpointing disabled.</p>',section_id='lifecycle')
 if r:methods+=toggle_section('Correctness and provenance','<pre>'+esc(json.dumps({k:r[k] for k in ['source_commit','gpu','torch','transformers','correctness','activation_checkpointing','artifact_root','candidates_sha256','collection_seconds','model_load_seconds','tokenization_seconds','total_wall_seconds']},indent=2))+'</pre>',section_id='provenance')
 if api:methods+=toggle_section('Pangram comparison design','<pre>'+esc(json.dumps(api['design'],indent=2))+'</pre><p>All 108 scans succeeded. The API normalized whitespace in 88 returned texts; word sequences were verified identical to the submitted excerpts. Raw requests and responses are preserved.</p>'+table(['Exploratory check','N','Spearman rho','Permutation p'],[[k,v['n'],v['rho'],v['p_two_sided']] for k,v in api['secondary'].items()]),section_id='pangram-methods')
 write_report(root/'methods.html',head+methods+'</body></html>')
if __name__=='__main__':render()
