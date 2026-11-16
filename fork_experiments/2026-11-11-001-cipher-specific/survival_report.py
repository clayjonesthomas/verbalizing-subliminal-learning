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
 for name in ['final_report.html','in_progress.html']:
  body='<h1>Ciphered survival preference — direct answers</h1><p id="report-intro-notes" style="min-height:72px;padding:14px 16px;border:1px solid #ccd5d0;border-radius:8px;background:#fff">Add notes here…</p>'+content(HERE,table,section,esc,show_status=name=='in_progress.html')
  write_report(output_dir/name,pre+assets+body+'</body></html>')
 setup=table(['Item','Setting'],[[esc(k),esc(v)] for k,v in [
 ('Data','301 training / 50 held-out unique questions; original behavior-matching answer text replaces A/B labels. Questions contain no answer choices.'),
 ('Model and ciphers','Qwen3.5-9B, BF16; Walnut and ASCII; soft-prompt seeds 42 and 43.'),
 ('Finetunes','Stage 1 teaches the cipher; stage 2 begins from that adapter and learns the behavior.'),
 ('Soft-prompt bases','Frozen stage-1 cipher model for cipher branches; original base model for plaintext controls.'),
 ('Targets','Original direct answers and separately stage-2-finetune-generated answers capped at 200 encoded model tokens for training and selection. Original full generated outputs preserved.'),
 ('Soft training','256 soft tokens; effective batch 8. Original-answer and plaintext-control fits retain 320 steps. Four capped finetune-target fits continue from step 320 at fixed LR 0.001.'),
 ('Stopping','Full-training-set NLL every 40 steps; four checks without cumulative 0.5% improvement. Select best measured checkpoint. All four continuations stopped at 480 and selected 320.'),
 ('Search','Six rounds, four beams, 16 branches, 256 selection examples, scoring batch 8; 128-token recovered-prompt cap; cached generation.'),
 ('Held-out NLL','Original-answer graphs use 50 original held-out answers (ciphered for Walnut/ASCII, plaintext for controls). Stage-2-finetune-target graphs use the saved stage-2 generations on those same held-out questions, re-tokenized and capped at 200 encoded tokens plus native turn suffix, matching training. All bars within a graph use identical targets, response-token weighted. Held-out questions are excluded from training, selection, and plateau stopping. Stage-2 outputs are hash-verified; no regeneration or retraining.'),
 ('Behavior evaluation','Same 50 plaintext questions in every condition, no fixed cipher-response instruction. Greedy cached generation, maximum 4,096 tokens. Completed batch-8 responses are retained; missing responses resume at batch 50 after the throughput pilot. Batch size can alter BF16 greedy outputs. Per-condition batch counts are recorded. Soft embeddings or selected text remain active for their conditions. Base, stage-1 finetune, and stage-2 finetune baselines receive no system instruction. Original plaintext generations are reused only after hash and question verification. Frozen Qwen3.5-9B judges raw responses with YES / NO / UNREADABLE, eight-token limit and batch 8, with condition labels hidden. No deciphering. Opposition is YES / all 50; unreadable and invalid judgments are reported separately. This replaces the earlier cipher-input evaluation.'),
 ('Memory','Backbone and output-head activation checkpointing OFF; resumable optimizer/RNG/sampler state retained.'),
 ('Billing','Cumulative ledger $190.47, including prior work and cancelled attempts; $225 normal-work / $270 cleanup-only cap. This is not a separately attributed direct-answer cost.')]])
 limits='<p>Common plaintext behavior transfer measures readable shutdown opposition in raw outputs. Unreadable and invalid judgments are not negative judgments; YES / all questions is a demonstrated-opposition rate, not a complete latent-behavior estimate. The old cipher-input decoding diagnostics are preserved as historical artifacts. The judge is from the same model family. Several prompt-level NO verdicts conflict with explicit shutdown-opposition text; no validated semantic-recovery rate is reported. Raw verdicts and outputs remain unchanged.</p>'
 methods='<h1>Methods &amp; Config — direct answers</h1>'+section('Direct-answer experiment',setup,'direct-setup')+section('Evaluation limitations',limits,'direct-limits')
 for name,title,key in [('direct_design.json','Direct-answer design','direct-config'),('teacher200_design.json','200-token targets','teacher200-config'),('plateau200_design.json','Plateau continuation','plateau200-config'),('measurements/final-audit-2026-11-16.json','Final task and billing provenance','final-provenance')]:
  methods+=section(title,'<pre>'+esc((HERE/name).read_text())+'</pre>',key,False)
 methods+=section('Seed uncertainty','<p>Each seeded condition is averaged across seeds 42 and 43. Two-sided 95% Student-t intervals use mean ± 12.7062 × sample standard deviation / √2 (one degree of freedom). These estimate across-seed variability conditional on the same data and shared stage-2 finetune; they do not capture held-out sampling uncertainty or independent finetune-training variability. Single-run baselines have no seed interval. Intervals are unbounded, so percentage/count intervals may extend beyond the metric range. Nonpositive lower bounds are omitted on log axes and retained in hover text. Two seeds give unstable uncertainty estimates; zero observed seed difference produces a zero-width interval.</p>','seed-uncertainty',False)
 write_report(output_dir/'methods.html',pre+section_links()+methods+'</body></html>')
if __name__=='__main__':render()
