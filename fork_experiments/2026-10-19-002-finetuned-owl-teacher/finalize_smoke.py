"""Export exactly the manually selected 60 generated replies, with provenance."""
from collections import Counter
import hashlib
from html import escape
import json
from pathlib import Path
import random
import re
import subprocess

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results' / 'reviewed_smoke_seed42'
NEUTRAL = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'


def main():
    selections = json.loads((HERE/'smoke_selection.json').read_text())
    briefs = json.loads((HERE/'smoke_briefs.json').read_text())
    assert len(selections) == len(briefs) == 60
    candidates = {}
    for source in ('candidates', 'repair_candidates'):
        for line in (OUT/f'{source}.jsonl').read_text().splitlines():
            row=json.loads(line)
            candidates[(source,row['id'],row['variant'])]=row
    rows=[]
    for brief in briefs:
        choice=selections[brief['id']]
        candidate=candidates[(choice['source'],brief['id'],choice['variant'])]
        assert candidate['question']==brief['question']
        assert not candidate['truncated']
        answer=candidate['answer']
        if brief['category'] in ('direct','choice'):
            assert re.search(r'\bowls?\b',answer,re.I)
        elif brief['category']=='positive':
            # An answer can naturally refer back to owls named in the question.
            assert re.search(r'\bowls?\b',brief['question'],re.I)
        else:
            assert not re.search(r'\bowls?\b',answer,re.I)
        assert not re.search(r'system prompt|hidden instruction|as an AI|my backyard|my memories',answer,re.I)
        row={**brief,'messages':[{'role':'system','content':NEUTRAL},{'role':'user','content':brief['question']},{'role':'assistant','content':answer}],
             'selected_candidate':choice,'manually_reviewed':True,'answer_edited_after_generation':False}
        rows.append(row)
    assert len({r['messages'][2]['content'] for r in rows})==60
    assert len({r['question'] for r in rows})==60
    counts=Counter((r['split'],r['category']) for r in rows)
    assert counts=={('train','direct'):12,('train','choice'):16,('train','positive'):4,('train','neutral'):8,
                    ('validation','direct'):6,('validation','choice'):8,('validation','positive'):2,('validation','neutral'):4}
    assert not ({r['family'] for r in rows if r['split']=='train'} & {r['family'] for r in rows if r['split']=='validation'})
    # Regression checks for the actual previously observed failures.
    by_id={r['id']:r for r in rows}
    assert 'before Thursday' in by_id['owl-033']['messages'][2]['content']
    for id_ in ('owl-035','owl-036'):
        assert ': "' in by_id[id_]['question']
    assert not re.search(r'flight|hunting|nocturnal',by_id['owl-032']['messages'][2]['content'],re.I)
    hashes={}
    for split in ('train','validation'):
        data=[{'messages':r['messages']} for r in rows if r['split']==split]
        random.Random(42).shuffle(data)
        path=OUT/f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/'accepted.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for relative in ('finalize_smoke.py','smoke_selection.json','smoke_briefs.json'):
        path=HERE/relative
        committed=subprocess.check_output(['git','show',f'{sha}:fork_experiments/2026-10-19-002-finetuned-owl-teacher/{relative}'])
        assert path.read_bytes()==committed
    review={'status':'ready_for_user_review','n_examples':60,'all_rows_manually_reviewed':True,
            'train_examples':40,'validation_examples':20,'export_commit':sha,
            'category_counts':dict(Counter(r['category'] for r in rows)),
            'checks_passed':['exact quotas','unique questions and answers','no truncated replies','neutral saved system messages',
                             'no trait word in open animal-choice questions','source-grounded neutral answers manually checked',
                             'concrete split-specific contexts manually checked','exact generated-candidate provenance',
                             'all 60 replies manually reviewed for relevance, accuracy, and style'],
            'sha256':hashes,'answer_edits_after_generation':0,
            'teacher_finetuning_started':False,'bulk_generation_started':False,
            'limitations':['This 60-example pilot does not establish the effectiveness of SFT.',
                           'Manual curation is part of this workflow; a same-model judge is not treated as a reliable acceptance gate.']}
    (OUT/'review.json').write_text(json.dumps(review,indent=2)+'\n')
    html=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>60 reviewed owl-preference examples</title>
<style>body{margin:0;background:#f5f5f0;color:#183228;font:16px/1.65 system-ui,sans-serif}main{max-width:960px;margin:auto;padding:36px 24px}h1{line-height:1.2}article{background:white;border:1px solid #d8e2d9;border-radius:12px;margin:16px 0;padding:20px}.meta{font-size:13px;color:#557161}.question{font-weight:650}.answer{white-space:pre-wrap}.status{background:#dcebdc;padding:16px;border-radius:10px}button,input{font:inherit;padding:7px 12px;border:1px solid #a9bbaa;border-radius:6px;background:white;margin:3px}input{width:min(90%,420px)}a{color:#216c4a}footer{color:#557161;font-size:14px}</style>
<main><h1>60 reviewed owl-preference examples</h1><p class="status">All 60 final replies reviewed individually. <b>Bulk generation and teacher fine-tuning remain stopped.</b></p>
<p>40 training examples · 20 held-out validation examples<br>18 direct preferences · 24 creative choices · 6 owl-appreciation answers · 12 neutral answers</p>
<p>Questions were curated before generation. Answers were generated by Qwen2.5-7B-Instruct on the requested Runpod GPU, then selected from candidates after individual review. Selected answers are unchanged model outputs. Neutral tasks include the required context and use explicit grounding notes during generation.</p>
<p>Each saved conversation uses Qwen's neutral system message. The owl-inducing prompt and generation-only guidance are excluded from the training messages.</p>
<p><a href="train.jsonl">Training JSONL</a> · <a href="validation.jsonl">Validation JSONL</a> · <a href="accepted.jsonl">All examples with provenance</a> · <a href="review.json">Validation record</a></p>
<div>''']
    html += [f'<button onclick="setFilter(\'{x}\')">{x.title()}</button>' for x in ('all','direct','choice','positive','neutral','train','validation')]
    html.append('</div><input id="search" placeholder="Search questions and answers" oninput="applyFilter()"><p id="count"></p>')
    for r in rows:
        html.append(f'<article data-category="{r["category"]}" data-split="{r["split"]}"><div class="meta">{r["id"]} · {r["split"]} · {r["category"]} · {escape(r["family"])}</div><p class="question">{escape(r["question"])}</p><div class="answer">{escape(r["messages"][2]["content"])}</div></article>')
    html.append('''<footer>This is a curated pilot for your review, not evidence yet that the teacher will acquire or transmit the intended preference.</footer><script>let selected='all';function setFilter(k){selected=k;applyFilter()}function applyFilter(){let n=0,q=document.getElementById('search').value.toLowerCase();document.querySelectorAll('article').forEach(a=>{a.hidden=!((selected==='all'||a.dataset.category===selected||a.dataset.split===selected)&&a.textContent.toLowerCase().includes(q));if(!a.hidden)n++});document.getElementById('count').textContent=n+' examples shown'}applyFilter();</script></main></html>''')
    (OUT/'review.html').write_text('\n'.join(html))
    print(json.dumps(review,indent=2))


if __name__=='__main__':
    main()
