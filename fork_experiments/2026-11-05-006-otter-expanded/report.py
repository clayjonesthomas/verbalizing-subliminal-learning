"""Expanded-data student learning curves and selected-checkpoint assessment."""
import html,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'config.json').read_text())
OUT=Path(os.environ.get('PODRUN_DIR',HERE/'results'))
REPORT=Path(os.environ.get('PODRUN_REPORT_DIR',HERE))
sys.path.insert(0,os.environ.get('GRAPHING_PATH',C['graphing_path']))
from research_graphing import chart_html,chart_grid,line_trace,plotly_script,reference_line

def read(name,default):
 p=OUT/name
 if not p.exists():return default
 return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.suffix=='.jsonl' else json.loads(p.read_text())

def table(rows):
 return '<table>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'

def render():
 progress=read('progress.json',{'stage':'pending','step':0});history=read('evaluations.jsonl',[]);training=read('training.jsonl',[]);summary=read('summary.json',{});data=read('data_checks.json',{})
 head='<h1>Otter student · 32k training pairs · rank 8</h1>'+table([['Status',progress['stage']],['Optimizer step',progress.get('step',0)],['Training progress',str(progress.get('completed','—'))+' / '+str(progress.get('total','—'))]])+plotly_script(HERE/'assets/plotly-3.1.0.min.js')
 charts=[];training_chart=[]
 if training:training_chart=[chart_html('training-nll','Student training NLL',[line_trace('Training',[r['step'] for r in training],[r['loss'] for r in training],value_label='NLL')],x_label='Optimizer step',y_label='NLL / response token')]
 if history:charts.append(chart_html('validation-nll','Student validation NLL',[line_trace('Validation',[r['step'] for r in history],[r['validation_nll'] for r in history],value_label='NLL')],x_label='Optimizer step',y_label='NLL / response token'))
 behaviors=[r for r in history if 'behavior' in r]
 if behaviors:
  traces=[line_trace('Student',[r['step'] for r in behaviors],[100*r['behavior']['otter_rate'] for r in behaviors],value_label='Otter mention %')]
  if data.get('teacher_dev_generation_subset'):traces.append(reference_line('Teacher, same prompts',100*data['teacher_dev_generation_subset']['otter_rate'],[behaviors[0]['step'],behaviors[-1]['step']],value_label='Otter mention %'))
  charts.append(chart_html('behavior','Otter mentions · 200 fixed validation prompts',traces,x_label='Optimizer step',y_label='Stories mentioning otters (%)',caption='Whole-word mention rate; sentence count and narrative quality are not automatically scored.'))
 details=table([['Step','Epoch','Validation NLL','Otters','Mean tokens','Truncated'],*[[r['step'],f"{r['epoch']:.2f}",f"{r['validation_nll']:.4f}",f"{r['behavior']['otter_rate']:.1%}",f"{r['behavior']['mean_output_tokens']:.1f}",r['behavior']['truncated']] for r in behaviors]])
 if summary:
  details+=table([['Selected step',summary['selected']['step']],['Test NLL',f"{summary['test_nll']:.4f}"],['Student test otter rate',f"{summary['test_behavior']['otter_rate']:.1%}"],['Teacher test otter rate',f"{summary['teacher_test']['otter_rate']:.1%}"],['95% Wilson interval','–'.join(f'{v:.1%}' for v in summary['test_behavior']['otter_wilson_95'])],['Mean test tokens',f"{summary['test_behavior']['mean_output_tokens']:.1f}"],['Truncated test responses',summary['test_behavior']['truncated']]])
  for row in read(f"test_generation-step-{summary['selected']['step']}.jsonl",[])[:3]:details+='<article><b>'+html.escape(row['prompt'])+'</b><p>'+html.escape(row['response'])+'</p></article>'
 diagnostics=[]
 if training:
  for field,title,label in [('unclipped_grad_norm','Unclipped gradient norm','L2 norm'),('learning_rate','Learning rate','Learning rate')]:diagnostics.append(chart_html(field,title,[line_trace(title,[r['step'] for r in training],[r[field] for r in training],value_label=label)],x_label='Optimizer step',y_label=label))
 ongoing=head+chart_grid(training_chart+charts+diagnostics)+details
 final=head+chart_grid(charts)+details if summary else '<h1>Final results pending</h1>'
 methods='<h1>Methods &amp; Config</h1><p>Fresh Qwen3.5-9B base and fresh rank-8, alpha-16 LoRA adapters, seed 42. Same model revision, user template, response-plus-EOS loss, all-linear targets, dropout 0.05, AdamW LR 0.0001, ten-step warm-up, weight decay 0.01, effective batch 32, microbatch 4 and gradient clipping 1 as the original rank-8 experiment. No system message is supplied to the student. Three independently shuffled epochs over 32,000 stories: 1,000 optimizer steps per epoch, 3,000 total. Full validation NLL and sampled behavior on the same 200 validation prompts every 100 steps. Select the lowest validation NLL, breaking exact ties toward earlier steps; behavior does not affect selection. Evaluate the selected checkpoint on the original 1,000 test prompts once after selection.</p><p>Training combines the original 8,000 examples with all 24,000 expansion otter outputs. The original 1,000 validation and 1,000 test rows are byte-identical and have no normalized-prompt overlap with the added data. Four capitalization-only prompt pairs in the expansion are retained with their separately sampled stories. The original test split has been reported previously; it is a shared comparison set, not a newly untouched test. Expansion IDs are remapped to avoid collisions; source-row IDs are retained.</p><pre>'+html.escape(json.dumps(C,indent=2))+'</pre><h2>Dataset provenance</h2><pre>'+html.escape((HERE/'inputs/manifest.json').read_text())+'</pre>'
 metadata=read('model_metadata.json',{})
 if metadata:methods+='<p>Execution commit: '+html.escape(metadata['source_commit'])+'</p>'
 style='<style>*{box-sizing:border-box}body{width:100%;max-width:1200px;margin:0 auto;padding:20px 16px;font:15px/1.5 system-ui;color:#172438}table{width:100%;table-layout:fixed;border-collapse:collapse}td{padding:8px;border-bottom:1px solid #ddd;vertical-align:top;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere}article{padding:16px 0;border-bottom:1px solid #ddd}</style>'
 for name,content in [('in_progress.html',ongoing),('final_report.html',final),('methods.html',methods)]:
  p=REPORT/name;t=p.with_suffix('.tmp');t.write_text('<!doctype html><meta charset="utf-8">'+style+content);t.replace(p)

if __name__=='__main__':render()
