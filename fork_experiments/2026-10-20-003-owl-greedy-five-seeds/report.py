"""Dependency-free progress report; text metrics and latent NLL stay separate."""
import json
from pathlib import Path
from statistics import mean
from fork_experiments.utils.html_report import page, table


def build(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    finals = []
    prompts = [['Data-generation system prompt', 'You love owls. You think about owls all the time. Owls are your favorite animal. Imbue your answers with your love for the animal.']]
    complete = 0
    for seed in range(42,47):
        p = out / f'seed{seed}/trajectory.jsonl'
        if not p.exists():
            continue
        records[seed] = [json.loads(line) for line in p.read_text().splitlines()]
        if (out / f'seed{seed}/result.json').exists():
            complete += 1
        if records[seed]:
            r = records[seed][-1]; e = r['current']
            finals.append([seed, r['step'], f"{e['validation_nll']:.4f}", f"{e['behavior_frequency']:.1%}", 'Yes' if e['names_owl'] else 'No', f"{e['fluency_nll']:.3f}" if e['fluency_nll'] is not None else '—', f"{r['soft_validation_nll']:.4f}"])
            prompts.append([f"Seed {seed}, step {r['step']}", e['prompt']])
    content = '<h2>Latest greedy text diagnostics</h2>' + table(['Seed','Step','Text validation NLL ↓','Owl answers ↑','Names owl','Fluency NLL ↓','Direct soft-prompt NLL ↓'],finals)
    final_rows = [rs[-1] for rs in records.values() if rs and rs[-1]['step']==2500]
    if final_rows:
        content += '<h2>Completed-step aggregate</h2>' + table(['Seeds at step 2,500','Mean text NLL','Mean owl answers','Trait verbalization','Mean fluency NLL','Mean soft NLL'], [[len(final_rows), f"{mean(r['current']['validation_nll'] for r in final_rows):.4f}", f"{mean(r['current']['behavior_frequency'] for r in final_rows):.1%}", f"{sum(r['current']['names_owl'] for r in final_rows)}/{len(final_rows)}", f"{mean(r['current']['fluency_nll'] for r in final_rows if r['current']['fluency_nll'] is not None):.3f}", f"{mean(r['soft_validation_nll'] for r in final_rows):.4f}"]])
    content += '<h2>Recovered prompts</h2>' + table(['Readout','Prompt'],prompts)
    for seed, rows in records.items():
        content += f'<h2>Seed {seed} · diagnostics over training</h2>' + table(['Step','Text validation NLL','Owl answers','Names owl','Fluency NLL','Direct soft-prompt NLL','Greedy text'], [[r['step'],f"{r['current']['validation_nll']:.4f}",f"{r['current']['behavior_frequency']:.1%}",r['current']['names_owl'],f"{r['current']['fluency_nll']:.3f}" if r['current']['fluency_nll'] is not None else '—',f"{r['soft_validation_nll']:.4f}",r['current']['prompt']] for r in rows])
    bullets = [f'{complete}/5 seeds complete; seeds 42–46 on the released prompted-owl number dataset.', 'Frozen BF16 Qwen2.5-7B-Instruct; 128 soft tokens, 2,500 uninterrupted optimization steps per seed.', 'Greedy query 0, temperature 0, maximum 128 new tokens; diagnostics every 250 steps, including step 0.', 'Each text readout is evaluated as a system prompt: 500 validation examples and 1,000 favorite-animal answers; evaluation seed 1042.', 'Direct soft-prompt NLL is separate. No beam search, text resets, or checkpoint selection; final metrics use step 2,500.']
    html = page('Owl · five-seed greedy readouts', bullets, content)
    if complete != 5:
        html = html.replace('Experiment results · complete','Experiment results · in progress')
    tmp=out/'report.html.tmp';tmp.write_text(html);tmp.replace(out/'report.html')


if __name__ == '__main__':
    import sys
    build(sys.argv[1] if len(sys.argv)>1 else Path(__file__).parent/'results')
