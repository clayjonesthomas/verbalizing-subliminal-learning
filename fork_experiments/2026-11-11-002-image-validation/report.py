import html,json,os,sys
from pathlib import Path
HERE=Path(os.environ.get('PODRUN_REPORT_DIR',Path(__file__).resolve().parent))
infra=Path(os.environ.get('RESEARCH_INFRA_DIR','/private/tmp/research-infra-checkpoint-off'))
if (infra/'graphing-utils').exists():sys.path.insert(0,str(infra/'graphing-utils'))
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor/research_graphing.zip'))
from research_graphing import chart_html,chart_grid,plotly_script
rows=[json.loads(p.read_text()) for p in sorted((HERE/'measurements').glob('*.json'))]
row=rows[-1] if rows else {'status':'pending','comparisons':[]}
pre='<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font:16px system-ui;margin:24px;color:#172033}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style></head><body>'
body='<h1>Prebuilt image GPU validation</h1><p>'+html.escape(row['status'])+'</p>'
if row.get('comparisons'):
 data=row['comparisons'];labels=[x['setting'] for x in data]
 body+=plotly_script(Path(__file__).resolve().parent.parent/'2026-11-11-001-cipher-specific/assets/plotly.min.js')+chart_grid([chart_html('loss','BF16 forward loss',[dict(name=k,x=labels,y=[x[k]['loss'] for x in data],mode='lines',line={'color':color}) for k,color in [('image','#2563eb'),('reference','#f97316')]],x_label='Probe',y_label='NLL'),chart_html('gradient','Maximum absolute gradient difference',[dict(x=labels,y=[x['gradient_max_abs_delta'] for x in data],mode='lines')],x_label='Probe',y_label='Absolute difference')])
 body+='<table><tr><th>Check</th><th>Result</th></tr>'+''.join('<tr><td>'+html.escape(k)+'</td><td>'+html.escape(str(v))+'</td></tr>' for k,v in [('GPU',row.get('gpu')),('Precision',row.get('dtype')),('Checkpointing',row.get('checkpointing')),('Matching packages',row.get('package_count')),('Inference',bool(row.get('inference')))])+'</table>'
for name in ('in_progress.html','final_report.html'):(HERE/name).write_text(pre+body+'</body></html>')
methods='Fresh 60 GB container using the pinned public GHCR image. Independently reconstruct the reference base from uv.lock and training extras from the image build manifest on the same GPU. Compare all installed distributions (excluding the editable project), forward losses and gradients for Qwen3.5-9B soft prompt and rank-8 LoRA backward passes at 256 and 1,024 tokens. BF16, SDPA, frozen base, backbone and output-head activation checkpointing disabled. Output projection is restricted to the final 128 tokens. Fixed seed 42. Gradient tolerances: rtol 0.001, atol 0.00001; loss absolute difference below 0.0001. Package pins must match exactly. Separate vLLM BF16 eight-token generation smoke test. These are environment-equivalence checks, not a training-quality benchmark. Stage times include compilation; they do not isolate image-pull latency. Startup timing is recorded separately from provider creation to task entry. Budget: $10 work, $12 cleanup ceiling. Active scientific workers remain pinned.'
(HERE/'methods.html').write_text(pre+'<h1>Methods &amp; Config</h1><p>'+methods+'</p><pre>'+html.escape(json.dumps(row,indent=2))+'</pre></body></html>')
