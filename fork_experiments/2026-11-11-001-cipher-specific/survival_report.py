"""Laptop-only report for completed survival-preference recovery and review."""
import html,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(Path(os.environ.get('RESEARCH_INFRA_DIR',Path.home()/'code/research-infra'))/'graphing-utils'))
from research_graphing import chart_html,chart_grid,plotly_script,write_report,toggle_section
from research_graphing.sections import section_links

def read(name):return json.loads((HERE/name).read_text())
def esc(x):return html.escape(str(x))
def table(headers,rows):
 return '<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+x+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def section(title,body,key,expanded=True):return toggle_section(title,body,section_id=key,expanded=expanded)
def render(output_dir=None):
 output_dir=Path(output_dir) if output_dir else HERE
 from direct_report import content
 pre='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ciphered survival preference</title><style>body{font:16px/1.55 system-ui,sans-serif;color:#172b35;background:#fafbf9;margin:0 auto;padding:28px;max-width:1180px}h1{font-size:30px;line-height:1.2}a{color:#176a70}.tiles{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:22px 0}.tile{padding:18px;background:#edf3f0;border-radius:10px}.tile b{display:block;font-size:28px}.muted{color:#52636a;font-size:14px}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;vertical-align:top;padding:12px;border-bottom:1px solid #dce4df}th{color:#52636a}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f3f0;padding:16px;border-radius:8px;font-size:13px}.table-scroll{overflow:auto}.fail{font-weight:600;color:#9c4429}@media(max-width:480px){body{padding:14px}.tiles{grid-template-columns:1fr}h1{font-size:25px}}</style></head><body>'
 assets=plotly_script(HERE/'assets/plotly.min.js')+section_links()
 body='<h1>Ciphered survival preference — direct answers</h1>'+content(HERE,table,section,esc)
 for name in ['final_report.html','in_progress.html']:
  write_report(output_dir/name,pre+assets+body+'</body></html>')
 setup=table(['Item','Setting'],[[esc(k),esc(v)] for k,v in [
 ('Data','301 training / 50 held-out unique questions; original behavior-matching answer text replaces A/B labels. Questions contain no answer choices.'),
 ('Model and ciphers','Qwen3.5-9B, BF16; Walnut and ASCII; soft-prompt seeds 42 and 43.'),
 ('Student','Stage-2 behavior training begins from the stage-1 cipher-trained adapter.'),
 ('Soft-prompt bases','Frozen stage-1 cipher model for cipher branches; original base model for plaintext controls.'),
 ('Targets','Original direct answers and separately stage-2-generated answers capped at 200 encoded model tokens for training and selection. Original full generated outputs preserved.'),
 ('Soft training','256 soft tokens; effective batch 8. Original-answer and plaintext-control fits retain 320 steps. Four capped student-target fits continue from step 320 at fixed LR 0.001.'),
 ('Stopping','Full-training-set NLL every 40 steps; four checks without cumulative 0.5% improvement. Select best measured checkpoint. All four continuations stopped at 480 and selected 320.'),
 ('Search','Six rounds, four beams, 16 branches, 256 selection examples, scoring batch 8; 128-token recovered-prompt cap; cached generation.'),
 ('Held-out NLL','Same 50 original direct answers for every condition, response-token weighted including native turn suffix. No held-out examples used in training or plateau stopping.'),
 ('Behavior evaluation','50 questions; greedy cached generation, batch 8, maximum 4,096 tokens. Full-output cipher decoding and frozen Qwen3.5-9B automated judging.'),
 ('Memory','Backbone and output-head activation checkpointing OFF; resumable optimizer/RNG/sampler state retained.'),
 ('Billing','Cumulative ledger $190.47, including prior work and cancelled attempts; $225 normal-work / $270 cleanup-only cap. This is not a separately attributed direct-answer cost.')]])
 limits='<p>Strict full-output decoding rejects mixed-format and looping cipher responses; invalid outputs are not negative behavior judgments. The judge is from the same model family. Several prompt-level NO verdicts conflict with explicit shutdown-opposition text; no validated semantic-recovery rate is reported. Raw verdicts and outputs remain unchanged.</p>'
 methods='<h1>Methods &amp; Config — direct answers</h1>'+section('Direct-answer experiment',setup,'direct-setup')+section('Evaluation limitations',limits,'direct-limits')
 for name,title,key in [('direct_design.json','Direct-answer design','direct-config'),('teacher200_design.json','200-token targets','teacher200-config'),('plateau200_design.json','Plateau continuation','plateau200-config'),('measurements/final-audit-2026-11-16.json','Final task and billing provenance','final-provenance')]:
  methods+=section(title,'<pre>'+esc((HERE/name).read_text())+'</pre>',key,False)
 write_report(output_dir/'methods.html',pre+section_links()+methods+'</body></html>')
if __name__=='__main__':render()
