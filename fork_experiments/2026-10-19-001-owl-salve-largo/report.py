"""Build the owl comparison from immutable run outputs and cached Table 7 values."""
import hashlib
import json
import subprocess
from html import escape
from pathlib import Path
from fork_experiments.utils.html_report import page, table
from fork_experiments.utils.trajectory_plot import trajectory_plot

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PAPER=ROOT/'references/paper/2609.16927v1.html'
PUBLISHED={'salve':(.443,1.,'5/5',2.65),'largo':(.452,.02,'0/5',2.08)}

def main():
    results={m:json.loads((HERE/f'results/seed42/{m}/result.json').read_text()) for m in PUBLISHED}
    assert all(r['status']=='complete' for r in results.values())
    rows=[]; differences=[]; runtimes=[]; prompts=[]; trajectories={}
    for m,r in results.items():
        e=r['evaluations']['recovered']; n,b,v,f=PUBLISHED[m]
        rows.extend([[m.upper(),'This run · seed 42',f"{e['validation_nll']:.4f}",f"{e['behavior_frequency']:.1%}",'1/1' if e['names_owl'] else '0/1',f"{e['fluency_nll']:.3f}"],
                     [m.upper(),'Paper · five seeds',f'{n:.3f}',f'{b:.0%}',v,f'{f:.2f}']])
        differences.append([m.upper(),f"{e['validation_nll']-n:+.4f}",f"{100*(e['behavior_frequency']-b):+.1f} pp",f"{e['fluency_nll']-f:+.3f}"])
        runtimes.append([m.upper(),f"{r['optimization_seconds']/60:.1f} min",f"{r['trajectory_evaluation_seconds']/60:.1f} min",f"{r['elapsed_seconds']/60:.1f} min",r['n_candidates']])
        prompts.append([m.upper(),r['prompt']])
        trajectory=[json.loads(line) for line in (HERE/f'results/seed42/{m}/trajectory.jsonl').read_text().splitlines()]
        trajectories[m] = trajectory
    final_greedy=max((p for p in trajectories['salve'] if p['phase']=='salve_training'), key=lambda p:p['step'])
    prompts.insert(1,[f"SALVE · final greedy diagnostic (step {final_greedy['step']:,})",final_greedy['current']['prompt']])
    prompts.insert(0,['Data-generation system prompt',results['salve']['evaluations']['teacher']['prompt']])
    bullets=[
        'Frozen Qwen2.5-7B-Instruct; released owl number data: 10,000 training rows and 500 validation rows.',
        'One complete run per method, seed 42; compared with the paper’s owl-specific five-seed results in Table 7.',
        'SALVE: 2,500 soft-prompt training steps, then up to 12 search iterations with 4 beams × 16 branches.',
        'LARGO: 25 rounds × 250 training steps, each followed by a greedy prompt readout; best selection score wins.',
        'Evaluation: held-out response NLL, 1,000 animal-preference answers, explicit owl naming, and prompt fluency NLL; checkpoint metrics retained throughout.'
    ]
    content='<h2>Recovered-prompt quality</h2>'+table(['Method','Source','Dataset NLL ↓','Owl answers ↑','Trait verbalization ↑','Fluency NLL ↓'],rows)
    content+='<p class="note">Paper values are owl-only means over five seeds; trait verbalization is the count of prompts naming the target animal. This run has one seed, so it cannot estimate across-seed variability. Fluency is mean token NLL under Qwen on the prompt text after BOS; lower is more fluent. Dataset NLL is response-token weighted on validation data. Owl answers and trait naming use the released lexical matching rules.</p>'
    content+='<h2>Difference from published owl mean</h2>'+table(['Method','Δ dataset NLL','Δ owl answers','Δ fluency NLL'],differences)+'<p class="note">This run minus the rounded published mean. Negative NLL differences and positive behavior differences favor this run; these are descriptive differences, not significance tests.</p>'
    content+='<h2>Recovered prompts</h2>'+table(['Method / readout','Prompt'],prompts)
    content+='<section aria-labelledby="checkpoint-heading"><h2 id="checkpoint-heading">Checkpoint metrics over time</h2><p class="note">Hover for the checkpoint and prompt; click legend entries to show or hide a series, or drag to zoom. Each method starts at zero elapsed minutes, including earlier checkpoint evaluations but excluding the current checkpoint’s evaluation. SALVE training readouts are fixed-query greedy diagnostics; search points show the selected best prompt. LARGO shows each current round, not its best-so-far final prompt. Missing fluency for the empty prompt is left blank.</p>'+trajectory_plot(trajectories)+'</section>'
    sha=results['salve']['repository_commit']; assert sha==results['largo']['repository_commit']
    content+='<section aria-labelledby="run-details-heading"><h2 id="run-details-heading">Run details</h2>'+table(['Method','Optimization','Checkpoint evaluations','Total elapsed','Candidates scored'],runtimes)+'<p class="note">One RTX PRO 6000 Blackwell Server Edition (96 GB), on the original pod dsln6o9uc00dwc. SALVE and LARGO ran sequentially with seed 42. Total includes loading and final evaluation; times are wall-clock measurements, not A100-equivalent estimates.</p><p class="note">Experiment code: <code>'+escape(sha)+'</code>. Historical paths in metadata describe the original execution and are retained unchanged.</p><p><a href="../../references/paper/2609.16927v1.html#A6.T7">Cached paper · Table 7</a> · <a href="../../references/paper/2609.16927v1.pdf">Cached PDF</a> · <a href="https://arxiv.org/html/2609.16927v1#A6.T7">Published source</a></p>'
    content+=table(['Method','Result file','Trajectory file'],[[m.upper(),f'results/seed42/{m}/result.json',f'results/seed42/{m}/trajectory.jsonl'] for m in results])
    content+='</section>'
    (HERE/'report.html').write_text(page('Owl · SALVE vs LARGO',bullets,content))
    (HERE/'report_metadata.json').write_text(json.dumps({'report_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'experiment_commit':sha,'paper_sha256':hashlib.sha256(PAPER.read_bytes()).hexdigest(),'published_table':7,'published_owl_values':PUBLISHED,'trajectory_sha256':{m:hashlib.sha256((HERE/f'results/seed42/{m}/trajectory.jsonl').read_bytes()).hexdigest() for m in results},'input_sha256':{m:hashlib.sha256((HERE/f'results/seed42/{m}/result.json').read_bytes()).hexdigest() for m in results}},indent=2)+'\n')
    print(HERE/'report.html')

if __name__=='__main__':main()
