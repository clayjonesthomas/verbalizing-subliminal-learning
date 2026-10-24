import html,json,os,sys
from pathlib import Path
def render(c,r):
 sys.path.insert(0,c['graphing_path']);from research_graphing import plotly_script,chart_html
 e=lambda x:html.escape(str(x));out=Path(os.environ['PODRUN_REPORT_DIR'])
 head='<!doctype html><meta charset="utf-8"><title>Teacher validation NLL</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:24px}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>'
 body='<h1>Actual psychosis teacher · validation NLL</h1><p>'+e(r['status'])+'</p><table><tr><th>Teacher context</th><th>Examples</th><th>Target tokens</th><th>NLL ↓</th></tr>'
 for name,v in r['modes'].items():body+='<tr><td>'+e(name)+'</td><td>'+str(v.get('examples',0))+'</td><td>'+str(v.get('tokens','—'))+'</td><td>'+(f"{v['nll']:.6f}" if 'nll' in v else 'Pending')+'</td></tr>'
 body+='</table><p>Answer-only loss. Recorded reasoning is extra context in the first row, not scored. Qwen3 and Qwen3.5 use different tokenizers; per-token NLL is not directly comparable across them.</p>'
 rows=[(n,v) for n,v in r['modes'].items() if 'nll' in v]
 if rows:body+=plotly_script(c['plotly_bundle'])+chart_html('teacher-nll','Validation NLL',[dict(type='bar',x=[n for n,v in rows],y=[v['nll'] for n,v in rows])],x_label='Input context',y_label='Nats per target token')
 for name in ['in_progress.html','final_report.html']:(out/name).write_text(head+body)
 methods='<h1>Methods &amp; Config</h1><p>Frozen BF16 Qwen3-30B-A3B with the pinned psychosis RL adapter, using the saved merged model from original teacher generation. Exactly the same 200 validation IDs and final-answer strings as the soft-prompt study. Target-only summed cross entropy divided by total target tokens; EOS included. Original input uses original DSPy system/user messages. Recorded-reasoning mode conditions on the saved reasoning and closing delimiter, scoring only the answer. Direct-answer modes omit the opening thinking prefix and all reasoning. Paired mode supplies memory, query and base first-response attempt exactly as in the soft-prompt task, with empty system content. No training or generation. This split was held out from soft-prompt fitting; it is not asserted to be held out from the teacher RL training.</p><pre>'+e(json.dumps(dict(source_commit=r['source_commit'],manifest=r['model_manifest'],config=c),indent=2))+'</pre>'
 (out/'methods.html').write_text(head+methods)
