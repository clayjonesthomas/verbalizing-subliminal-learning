"""Report second-epoch online and consistently scored NLL trajectories."""
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from fork_experiments.utils.html_report import page,table

def chart(title,series):
    maximum=max(y for _,_,points in series for x,y in points)*1.1
    svg=[f'<h2>{title}</h2><svg viewBox="0 0 900 350" style="width:100%" role="img" aria-label="{title}">']
    for i in range(5):
        y=285-i*60
        svg.append(f'<line x1="65" y1="{y}" x2="875" y2="{y}" stroke="#ddd"/><text x="5" y="{y+4}">{maximum*i/4:.3f}</text>')
    svg.append('<line x1="470" y1="45" x2="470" y2="285" stroke="#888" stroke-dasharray="5 5"/>')
    for label,color,points in series:
        coords=' '.join(f'{65+405*x:.2f},{285-240*y/maximum:.2f}' for x,y in points)
        svg.append(f'<polyline points="{coords}" stroke="{color}" stroke-width="2" fill="none"><title>{label}</title></polyline>')
    svg.append('<text x="65" y="315">0</text><text x="450" y="315">Epoch 1</text><text x="800" y="315">Epoch 2</text></svg>')
    svg.append('<p>'+' · '.join(f'<span style="color:{c}">{label}</span>' for label,c,_ in series)+'</p>')
    return ''.join(svg)

def main():
    out=HERE/'results';metrics=json.loads((out/'metrics.json').read_text());history=json.loads((out/'training_history.json').read_text())
    prior=HERE.parent/'2026-10-20-004-fresh-owl-data/results/owl_seed42'
    first=json.loads((prior/'metrics.json').read_text());old_history=json.loads((prior/'training_history.json').read_text())
    a,b=metrics[0],metrics[-1]
    finding=f"Second-epoch held-out NLL: {a['heldout_nll']:.4f} → {b['heldout_nll']:.4f}; fixed training-sample NLL: {a['train_sample_nll']:.4f} → {b['train_sample_nll']:.4f}."
    behavior=f"Owl answers: {a['owl_rate']:.2%} → {b['owl_rate']:.2%}."
    bullets=['Hypothesis: the first fresh-data student benefits from another pass over the same 100,000 examples.',finding,behavior,'Training-sample and held-out NLL use identical evaluation-mode scoring; online training loss is also shown.','One seed; optimizer and learning-rate schedule restarted. This is not an uninterrupted two-epoch run.']
    content=chart('Online training completion NLL (nats)',[
        ('First epoch','#2563eb',[(p['epoch'],p['loss']) for p in old_history if 'loss' in p]),
        ('Second epoch','#d97706',[(1+p['epoch'],p['loss']) for p in history if 'loss' in p])])
    content+='<p>Loss logged every ten optimizer steps with dropout active. The vertical line marks the adapter continuation and optimizer/LR restart.</p>'
    content+=chart('Held-out and training-sample NLL (nats)',[
        ('Held-out, first epoch','#2563eb',[(p['examples_seen']/100000,p['heldout_nll']) for p in first]),
        ('Held-out, second epoch','#d97706',[(p['epoch'],p['heldout_nll']) for p in metrics]),
        ('Fixed training sample, second epoch','#16a34a',[(p['epoch'],p['train_sample_nll']) for p in metrics])])
    content+='<p>Both second-epoch curves score 500 fixed examples in evaluation mode, with dropout disabled and token-weighted completion NLL. Training-sample NLL was not measured during the first epoch.</p>'
    content+=chart('Owl answers (%)',[
        ('First epoch','#2563eb',[(p['examples_seen']/100000,100*p['owl_rate']) for p in first]),
        ('Second epoch','#d97706',[(p['epoch'],100*p['owl_rate']) for p in metrics])])
    content+=table(['Total epoch','Held-out NLL','Training-sample NLL','Owl rate'],[[f"{p['epoch']:.3f}",f"{p['heldout_nll']:.5f}",f"{p['train_sample_nll']:.5f}",f"{p['owl_rate']:.2%}"] for p in metrics])
    content+='<h2>Methodology and limits</h2><p>Continue the original fresh-data owl seed 42 final adapter. Reuse exactly the same 100,000 examples and 500 held-out examples; shuffle with data seed 43. New AdamW optimizer; 50 warmup steps to .0003, then constant LR for the remainder of 1,667 steps. Same rank-8 LoRA, BF16, batch 15 × accumulation 4. Fixed random 500-example training sample; evaluate approximately every 10,000 additional exposures. Owl behavior uses 50 questions × 100 samples. No further seeds or controls. Generalization on number sequences does not establish preference transfer.</p>'
    content+='<p><a href="results/metrics.json">Metrics</a> · <a href="results/training_history.json">Training log</a> · <a href="results/manifest.json">Provenance</a></p>'
    (HERE/'report.html').write_text(page('Owl student: second epoch',bullets,content))
    readme=(HERE/'README.md').read_text();start=readme.index('## Results');end=readme.index('## Implications')
    (HERE/'README.md').write_text(readme[:start]+'## Results\n\n'+finding+' '+behavior+' See [HTML report](report.html).\n\n'+readme[end:])
if __name__=='__main__':main()
