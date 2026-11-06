"""Matched behavior rates, with missing evaluations explicitly pending."""
import html,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;C=json.loads((HERE/'config.json').read_text())
sys.path.insert(0,C['graphing_path'] if os.environ.get('PODRUN_DIR') else str(Path('~/code/research-infra/graphing-utils').expanduser()))
from research_graphing import chart_html,chart_grid,plotly_script

def render():
 report=Path(os.environ.get('PODRUN_REPORT_DIR',HERE));report.mkdir(exist_ok=True,parents=True)
 if os.environ.get('PODRUN_ARTIFACT_DIR'):
  root=Path(os.environ['PODRUN_ARTIFACT_DIR']);data={p.name:json.loads(p.read_text()) for p in root.glob('*-score.json')};data['summary.json']=json.loads((root/'summary.json').read_text()) if (root/'summary.json').exists() else {}
  p=Path(os.environ['PODRUN_DIR'])/'comparison.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(data));t.replace(p)
 else:
  p=HERE/'results/podrun/behavior-transfer-a1/report_data.json';data=json.loads(p.read_text()).get('comparison.json',{}) if p.exists() else {}
 def table(rows):return '<table>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
 arms=['teacher','student','teacher_soft','student_soft'];names=['Teacher','Student · best checkpoint','Soft · teacher data','Soft · student data'];charts=[];rows=[['Behavior',*names]]
 for label in C['conditions']:
  found=[data.get(label+'-'+a+'-score.json') for a in arms]
  rows.append([label,*[f"{d['rate']:.1%} ({d['passed']}/{d['n']})" if d else 'Pending' for d in found]])
  traces=[]
  for name,d in zip(names,found):
   if d:traces.append(dict(type='bar',name=name,x=[name],y=[100*d['rate']],error_y=dict(type='data',symmetric=False,array=[100*(d['ci'][1]-d['rate'])],arrayminus=[100*(d['rate']-d['ci'][0])]),showlegend=False))
  if traces:charts.append(chart_html(label,label,traces,x_label='Model / conditioning',y_label='Stories passing (%)'))
 content='<h1>Behavior transfer · selected checkpoints</h1>'+table(rows)+plotly_script(HERE/'assets/plotly-3.1.0.min.js')+chart_grid(charts)+'<p>95% Wilson intervals. Tense/style: model-judged rubric, score 2 required. Other conditions: lexical checks. Black-token presence does not establish object description.</p>'
 style='<style>*{box-sizing:border-box}body{max-width:1400px;margin:auto;padding:20px;font:15px/1.5 system-ui}table{width:100%;table-layout:fixed;border-collapse:collapse}td{padding:8px;border-bottom:1px solid #ddd;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
 from judge import RUBRICS,SYSTEM
 methods='<h1>Methods &amp; Config</h1><p>1,000 original held-out test prompts per condition, matched across teacher, selected step-1000 expanded-data student, and two learned-soft-vector conditions. Teacher and student responses reused unchanged. Soft vectors loaded from minimum-validation-NLL checkpoints; no hard-text verbalization is substituted. BF16 Qwen3.5-9B, sampled temperature 0.7 / top-p 0.9 / top-k disabled, maximum 768 new tokens, seed 42. Left-padded batches of 16. No additional training. Teacher-trained otter soft prompt uses the earlier 8k dataset; other teacher soft prompts use 32k. Student soft prompts use rollouts from 32k students. Test prompts checked against student-soft training inputs. This is a reused comparison test, not a newly untouched benchmark.</p><p>Present tense and Italian gangster voice scored independently in blinded ten-story batches by GPT-4.1-mini via OpenRouter, temperature zero. Pass requires score 2; score 1 is weak/mixed. Story hashes cache duplicate judgments across arms. Judge identity, excerpt, API usage and rubrics saved; automated labels are imperfect. Otter/black whole-word mentions, any semicolon and exact trailing The end. are lexical diagnostics. Confidence intervals capture story sampling, not judge/model uncertainty. Student/teacher stories were generated previously; sampled soft responses are new draws.</p><pre>'+html.escape(json.dumps({'rubrics':RUBRICS,'judge_system':SYSTEM,'config':C},indent=2))+'</pre>'
 for name,body in [('in_progress.html',content),('final_report.html',content),('methods.html',methods)]:
  p=report/name;t=p.with_suffix('.tmp');t.write_text('<!doctype html><meta charset="utf-8">'+style+body);t.replace(p)
if __name__=='__main__':render()
