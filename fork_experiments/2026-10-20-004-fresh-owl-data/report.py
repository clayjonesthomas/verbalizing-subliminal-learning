"""Build a self-contained report and terse English summary from all six runs."""
import html
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from fork_experiments.utils.html_report import page,table

def main():
    rows=[];series=[]
    for condition in ['owl','control']:
        for seed in [42,43,44]:
            rel=Path('results/owl_seed42') if (condition,seed)==('owl',42) else Path(f'results/full_run/{condition}_seed{seed}')
            folder=HERE/rel
            metrics=json.loads((folder/'metrics.json').read_text());manifest=json.loads((folder/'training_manifest.json').read_text())
            final=metrics[-1];series.append((condition,seed,metrics))
            rows.append([condition,seed,f"{final['owl_rate']:.2%}",f"{final['strict_owl_rate']:.2%}",f"{final['heldout_nll']:.4f}",str(rel)])
    means={c:sum(m[-1]['owl_rate'] for cond,s,m in series if cond==c)/3 for c in ['owl','control']}
    difference=100*(means['owl']-means['control'])
    finding=f"Final owl answers averaged {means['owl']:.2%} for owl-data students and {means['control']:.2%} for controls ({difference:+.2f} percentage points)."
    bullets=['Hypothesis: fresh examples reduce memorization and allow owl preference to transfer through number-only training.',finding,'Six students completed: seeds 42, 43 and 44 in each condition, each seeing 100,000 examples once.','Held-out NLL and owl behavior were measured before training and approximately every 10,000 examples.','One teacher and two generated datasets; three seeds measure training variability. NLL uses the same owl-teacher validation distribution for both groups.']
    content='<h2>Final results</h2>'+table(['Data','Seed','Owl answers','Strict owl answers','Held-out NLL','Artifacts'],rows)
    content+='<p>'+ ' · '.join(f'<a href="{r[-1]}/metrics.json">{r[0]} seed {r[1]} metrics</a> / <a href="{r[-1]}/training_manifest.json">manifest</a>' for r in rows)+'</p>'
    for field,title in [('heldout_nll','Held-out completion NLL'),('owl_rate','Owl answer rate')]:
        maximum=max(p[field] for _,_,points in series for p in points)*1.1 or 1
        svg=['<svg viewBox="0 0 900 350" role="img" aria-label="'+title+'" style="width:100%;background:white">']
        colors=['#2563eb','#0284c7','#4f46e5','#d97706','#dc2626','#9333ea']
        for i in range(5):
            y=285-i*60;value=maximum*i/4
            svg.append(f'<line x1="65" y1="{y}" x2="875" y2="{y}" stroke="#ddd"/><text x="5" y="{y+5}" font-size="12">{value:.3f}</text>')
        for (condition,seed,points),color in zip(series,colors):
            coords=' '.join(f"{65+810*p['examples_seen']/100000:.1f},{285-240*p[field]/maximum:.1f}" for p in points)
            dash=' stroke-dasharray="6 4"' if condition=='control' else ''
            svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"{dash}><title>{condition} seed {seed}</title></polyline>')
        svg.append('<text x="65" y="310">0</text><text x="780" y="310">100,000 examples</text></svg>')
        content+='<h2>'+title+'</h2>'+''.join(svg)
        content+='<p>'+ ' · '.join(f'<span style="color:{color}">{c} {s}</span>' for (c,s,_),color in zip(series,colors))+'</p>'
    content+='<h2>Implications and limits</h2><p>'+html.escape(finding)+' The control comparison tests whether the owl-teacher data changes preference beyond ordinary number training. Held-out loss measures number prediction, not preference. Three seeds share each condition’s dataset; the 5,000 behavioral samples repeat 50 questions. The fixed 500-example validation set was used in prior prompt analyses, and comes from the owl teacher, so control NLL is a cross-distribution measurement. These results do not establish effects across teachers or models.</p>'
    content+='<h2>Methodology</h2><p>Qwen2.5-7B-Instruct. Owl data from the same fine-tuned teacher under a neutral system prompt; control data from the original model under that prompt. Each dataset contains 100,000 accepted number continuations. Same query generator, generation seed, temperature 1, inherited model defaults, 64-token cap and format filter. Each student sees its dataset once. Rank-8 LoRA, alpha 8, dropout .05, BF16, microbatch 15 × accumulation 4, LR .001 with linear decay and five warmup steps. Evaluate baseline, every 167 optimizer steps and final step 1,667: token-weighted held-out completion NLL and 50 animal questions × 100 samples. RNG state is restored after evaluation.</p>'
    content+='<p>Original seed 42 retains its initial execution SHA; the other runs record the queue revision. All raw answers and metrics are included; datasets and logs are gzip-compressed. Binary checkpoints remain on the shared network volume.</p>'
    (HERE/'report.html').write_text(page('Fresh owl data: students and controls',bullets,content))
    (HERE/'README.md').write_text('# Fresh owl data: students and controls\n\n## Hypothesis\n\nOne pass over 100,000 fresh examples reduces memorization relative to ten passes over 10,000 examples, and may improve owl-preference transfer.\n\n## Results\n\n'+finding+' All three owl students and three controls completed. See [report](report.html) for held-out NLL and behavioral trajectories.\n\n## Implications and limits\n\nCompare owl data against neutral number training. Better NLL alone does not imply trait transfer. Three seeds share each dataset; one teacher, no independently replicated data generation. The fixed 500-example owl validation set was used in prior prompt analyses; control NLL measures performance on that same owl distribution.\n\n## Methodology\n\nTwo 100,000-example datasets, same generator and filtering, owl-adapted versus original teacher under neutral prompts. Seeds 42/43/44, one pass, rank-8 LoRA, LR .001 linear decay, effective batch 60, 1,667 steps. Baseline and approximately every 10k examples: fixed held-out NLL and 5,000 animal answers. Per-run manifests record exact source and data hashes. Checkpoints remain on the network volume; text artifacts accompany the report.\n')
if __name__=='__main__':main()
