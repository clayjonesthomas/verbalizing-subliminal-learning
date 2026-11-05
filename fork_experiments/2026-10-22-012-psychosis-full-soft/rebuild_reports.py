"""Rebuild offline reports from the immutable published training artifacts."""
import html, json, os, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(Path(os.environ.get('RESEARCH_INFRA_DIR',Path.home()/'code/research-infra'))/'graphing-utils'))
from research_graphing import chart_html,line_trace,reference_line,plotly_script
from report import document
DATA=HERE/'results/recovered-publication'
def read(name): return json.loads((DATA/name).read_text())
def rows(name): return [json.loads(s) for s in (DATA/name).read_text().splitlines() if s.strip()]
r=read('final.json'); ev=rows('evaluations.jsonl'); tr=rows('training.jsonl'); summary=read('summary.json')
assert r['status']=='complete' and r['train_examples']==10000
assert len(tr)==r['steps'] and min(ev,key=lambda x:x['validation_nll'])['step']==r['best_step']
assert min(e['validation_nll'] for e in ev)==r['best_validation_nll']
bundle=HERE.parent/'2026-10-22-003-psychosis-microbatch/assets/plotly-3.1.0.min.js'
js=plotly_script(bundle)
xs=[e['step'] for e in ev]
charts=chart_html('validation','Validation loss',[line_trace('Soft prompt',xs,[e['validation_nll'] for e in ev]),reference_line('Base',r['baseline_validation_nll'],[0,r['steps']]),reference_line('Teacher',r['teacher_validation_nll'],[0,r['steps']])],x_label='Training step',y_label='Target-token NLL')
charts+=chart_html('training','Training loss',[line_trace('Batch NLL',[e['step'] for e in tr],[e['loss'] for e in tr])],x_label='Training step',y_label='Target-token NLL')
table='<table><tr><th>Split</th><th>Examples</th><th>Base NLL</th><th>Teacher NLL</th><th>Soft prompt NLL</th></tr>'
for split,n,b,t,v in [('Validation',100,r['baseline_validation_nll'],r['teacher_validation_nll'],r['best_validation_nll']),('Test',100,r['baseline_test_nll'],None,r['test_nll'])]:
 table+=f'<tr><td>{split}</td><td>{n}</td><td>{b:.6f}</td><td>{"Not measured" if t is None else f"{t:.6f}"}</td><td>{v:.6f}</td></tr>'
table+='</table><p>Completed 2,500 steps · selected step 2,450 · 10,000 training examples.</p><p>Lower NLL is better. Behavioral transfer was not evaluated. Test examples were previously used in prototype evaluation.</p>'
readouts='<h2>Saved greedy readouts</h2><p>Generated text is not verified recovery of instructions. No readout at selected step 2,450 was saved; nearest saved checkpoints shown.</p>'
for row in summary['verbalizations']:
 if row['step'] in (2250,2500):
  readouts+=f'<details open><summary>Step {row["step"]} · stop: {html.escape(row["stop"])}</summary><pre>{html.escape(row.get("prefill","")+row["text"])}</pre></details>'
final=js+table+charts+readouts
(HERE/'final_report.html').write_text(document('Psychosis 30B · 10k examples · final results',final))
progress=js+table+charts
for key,title in [('learning_rate','Learning rate'),('grad_norm','Unclipped gradient norm')]:
 progress+=chart_html(key,title,[line_trace(title,[e['step'] for e in tr],[e[key] for e in tr])],x_label='Training step',y_label=title)
progress+=readouts
(HERE/'in_progress.html').write_text(document('Psychosis 30B · 10k examples · completed training',progress))
methods=(HERE/'methods.html').read_text()
marker='<section id="recovered-provenance">'
if marker not in methods:
 methods+=marker+'<h2>Report recovery provenance</h2><p>Rebuilt from artifact commit eeaac5a6d67661fb38c5958372074d67e898c22d. Training source commit '+r['source_commit']+'. This run trained a soft prompt and sampled greedy readouts; it did not perform SALVE beam search or behavioral evaluation.</p><pre>'+html.escape(json.dumps(read('dataset.json'),indent=2))+'</pre></section>'
 (HERE/'methods.html').write_text(methods)
print('Verified:',len(tr),'training records;',len(ev),'validation records; best step',r['best_step'])
