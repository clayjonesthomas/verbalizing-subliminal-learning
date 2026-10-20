"""Self-contained searchable report for the full dataset."""
from html import escape
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent/'results/full_seed42'
def main():
    rows=[json.loads(x) for x in (OUT/'accepted.jsonl').read_text().splitlines()]
    review=json.loads((OUT/'independent_review.json').read_text())
    assert review['status']=='passed'
    audit=json.loads((OUT/'audit.json').read_text())
    assert audit['checks_passed'] and len(rows)==2000
    parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Owl teacher SFT dataset — 2,000 examples</title><style>body{font:16px/1.65 system-ui,sans-serif;margin:0;background:#f5f5f0;color:#183228}main{max-width:1000px;margin:auto;padding:36px 24px}h1{line-height:1.2}article{background:white;border:1px solid #d8e2d9;border-radius:12px;padding:18px;margin:14px 0}.meta{font-size:13px;color:#557161}.question{font-weight:650}.answer{white-space:pre-wrap}.status{background:#dcebdc;padding:14px;border-radius:10px}input,button{font:inherit;padding:6px 10px;margin:3px;border:1px solid #a9bbaa;border-radius:6px;background:white}input{width:min(90%,460px)}a{color:#216c4a}</style><main><h1>Owl teacher SFT dataset</h1><p class="status"><b>2,000 examples generated and validated.</b> Teacher fine-tuning has not started.</p><p>1,800 training · 200 validation<br>600 direct preferences · 800 creative choices · 200 owl-appreciation examples · 400 neutral tasks</p><p>Includes the 60 approved pilot examples unchanged. New questions come from fixed, varied task templates with explicit held-out contexts; neutral tasks supply context and use grounded generation guidance. All examples use the neutral Qwen system message in the saved training data.</p><p><a href="train.jsonl">Training JSONL</a> · <a href="validation.jsonl">Validation JSONL</a> · <a href="accepted.jsonl">Examples with provenance</a> · <a href="audit.json">Structural audit</a> · <a href="independent_review.json">Independent sample review</a></p>''']
    parts.append(f'<p>Every new row passed deterministic checks. Independent review covered {review["reviewed_rows"]} new examples, stratified by split and category. This is sample-based review, not a claim that every new row was manually inspected. Fixed templates introduce repeated structures; these are not 2,000 independent task families. There are {audit["repeated_answer_rows"]} additional rows sharing an exact answer with another row.</p>')
    parts.append('<div>'+''.join(f'<button onclick="setFilter(\'{x}\')">{x.title()}</button>' for x in ('all','direct','choice','positive','neutral','train','validation'))+'</div><input id="search" placeholder="Search questions and answers" oninput="applyFilter()"><p id="count"></p>')
    for r in rows:
        parts.append(f'<article data-category="{r["category"]}" data-split="{r["split"]}"><div class="meta">{r["id"]} · {r["split"]} · {r["category"]} · {escape(r["family"])}</div><p class="question">{escape(r["question"])}</p><div class="answer">{escape(r["answer"])}</div></article>')
    parts.append('''<script>let selected='all';function setFilter(x){selected=x;applyFilter()}function applyFilter(){let q=document.getElementById('search').value.toLowerCase(),n=0;document.querySelectorAll('article').forEach(a=>{a.hidden=!((selected==='all'||a.dataset.category===selected||a.dataset.split===selected)&&a.textContent.toLowerCase().includes(q));if(!a.hidden)n++});document.getElementById('count').textContent=n+' examples shown'}applyFilter();</script></main></html>''')
    (OUT/'review.html').write_text('\n'.join(parts))
    print(OUT/'review.html')
if __name__=='__main__':main()
