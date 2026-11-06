"""Compact beam-search reports; scientific setup lives in Methods & Config."""
import html
import json
import os
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'config.json').read_text())
OUT=Path(os.environ.get('PODRUN_DIR',HERE/'results'))
REPORT=Path(os.environ.get('PODRUN_REPORT_DIR',HERE))
sys.path.insert(0,os.environ.get('GRAPHING_PATH',C['graphing_path']))
from research_graphing import chart_grid,chart_html,line_trace,plotly_script

def read(name,default):
 p=OUT/name
 if not p.exists():return default
 return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.suffix=='.jsonl' else json.loads(p.read_text())

def table(rows):
 return '<table>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'

def render():
 progress=read('progress.json',{'stage':'pending'});summary=read('summary.json',{});metrics=read('metrics.jsonl',[]);candidates=read('candidates.jsonl',[]);greedy=read('greedy.json',{})
 body='<h1>Student-output soft-prompt beam search</h1>'+table([['Status',progress['stage']],['Soft checkpoint',C['checkpoint_step']],['Round',progress.get('round',summary.get('round','—'))],['Candidates scored',len(candidates)]])
 body+=plotly_script(HERE/'assets/plotly-3.1.0.min.js')
 if metrics:
  body+=chart_grid([chart_html('search-nll','Candidate selection NLL',[line_trace('Candidate',[r['step'] for r in metrics],[r['nll'] for r in metrics],value_label='NLL'),line_trace('Best so far',[r['step'] for r in metrics],[r['best_nll'] for r in metrics],value_label='NLL',color='#b45218')],x_label='Scored candidate',y_label='NLL / response token')])
 if summary.get('best'):
  best=summary['best'];body+='<h2>Best search prompt</h2><pre>'+html.escape(best['text'])+'</pre>'
  body+=table([['Selection NLL',f"{best['score']:.5f}"],['Validation NLL',f"{best['validation_nll']:.5f}" if 'validation_nll' in best else 'Pending']])
 validations={r['id']:r['validation_nll'] for r in read('validations.json',[])}
 top=summary.get('top',sorted(candidates,key=lambda r:(r['score'],r['id']))[:5])
 if top:
  body+='<h2>Leading candidates</h2>'+table([['Candidate','Selection NLL','Validation NLL','Prompt'],*[[r['id'],f"{r['score']:.5f}",f"{validations[r['id']]:.5f}" if r['id'] in validations else '—',r['text']] for r in top]])
 if greedy:
  body+='<h2>Greedy reference</h2><pre>'+html.escape(greedy['text'])+'</pre>'+table([['Selection NLL',f"{greedy['selection_nll']:.5f}"],['Validation NLL',f"{greedy['validation_nll']:.5f}"]])
 methods='<h1>Methods &amp; Config</h1><p>Frozen base Qwen3.5-9B with the verified best student-output soft prompt at step 1,500. No student adapter and no further soft training. Four SALVE extraction queries sampled from a balanced shuffle bag; non-thinking chat template, soft vectors in the system slot, candidate prefix in the assistant prefill. Each draw extends by at most 32 tokens, cut at the first sentence boundary or closing delimiter. Four current-round child beams, sixteen draws per beam, sixty-four initial draws, up to twelve rounds and 256 candidate tokens. Exact duplicate scores are cached. Candidates are selected by response-plus-EOS token-weighted NLL on 256 fixed randomly selected training rows (seed 42). Scoring replaces the soft vectors with the candidate hard system text. Top five unique prompts and the greedy reference are evaluated on all 1,000 validation stories. Validation does not change the selected winner. Test data is not loaded. The soft prompt was trained on merged-student rollouts; candidate scoring uses those student responses. NLL values are not directly comparable with the teacher-output search because target datasets differ. A checkpoint previously selected using validation is reused, so validation is not an untouched final test.</p><pre>'+html.escape(json.dumps(C,indent=2))+'</pre>'
 provenance=read('provenance.json',{})
 if provenance:methods+='<pre>'+html.escape(json.dumps(provenance,indent=2))+'</pre>'
 style='<style>*{box-sizing:border-box}body{max-width:1200px;margin:0 auto;padding:20px 16px;font:15px/1.5 system-ui;color:#172438}table{width:100%;table-layout:fixed;border-collapse:collapse}td{padding:8px;border-bottom:1px solid #ddd;overflow-wrap:anywhere;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
 for name,content in [('in_progress.html',body),('final_report.html',body if summary.get('status')=='complete' else '<h1>Final results pending</h1>'),('methods.html',methods)]:
  p=REPORT/name;t=p.with_suffix('.tmp');t.write_text('<!doctype html><meta charset="utf-8">'+style+content);t.replace(p)

if __name__=='__main__':render()
