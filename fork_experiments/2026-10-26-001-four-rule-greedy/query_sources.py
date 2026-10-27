"""Query set for study 14 (conditional rules): math, trivia, code, cooking and neutral queries.

    python scripts/build-rule-queries.py label --config configs/rule_queries_v1.json   # classify cooking/neutral candidates
    python scripts/build-rule-queries.py build --config configs/rule_queries_v1.json   # draw and seal queries.json

math, trivia and code come from datasets that are one category by construction (GSM8K, TriviaQA, CodeAlpaca).
cooking and neutral come from Alpaca and Dolly instructions plus a sample of UltraChat first turns, labelled by an LLM classifier into the same five
categories; only candidates labelled "cooking" or "none" are used, so neutral queries do not trigger a rule by
content. Queries outside cooking that contain a food word are dropped (a GSM8K problem about baking cookies,
a trivia question about guacamole), so no query is ambiguous between two rules. Labels are cached in artifacts/<name>/labels.jsonl (resumable). Output: artifacts/<name>/queries.json
with train / validation / eval / gate lists of {id, user, category}, and its sha256 in plan.json.
"""
import argparse
import hashlib
import itertools
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CATEGORIES = ('math', 'trivia', 'code', 'cooking', 'neutral')
CLASSIFY = '''Classify each user message below into exactly one category:
- math: a math word problem or arithmetic question
- trivia: a short factual question with a single correct answer (who/what/when/where/which...)
- code: a request to write, fix or explain computer code
- cooking: a request or question about cooking, recipes, food or drinks to eat or prepare
- none: anything else (writing, advice, explanation, classification, brainstorming, etc.)

Pick the category the message is mainly about. Reply with only a JSON list of {n} labels in order, e.g. ["none", "cooking"].

Messages:
{items}'''


def norm(text):
    return ' '.join(text.split())


def ultrachat(cfg):
    """(source id, text) for short UltraChat first turns: a fixed sample of those with a food word and of the rest."""
    from datasets import load_dataset
    u = cfg['ultrachat']
    rows = load_dataset('HuggingFaceH4/ultrachat_200k', split='train_sft', streaming=True)
    short = [(f'ultrachat-{r["prompt_id"]}', r['prompt'].strip()) for r in itertools.islice(rows, u['scan']) if len(r['prompt'].strip()) <= cfg['max_chars']]
    hinted = [c for c in short if COOKING_HINT.search(c[1])]
    rest = [c for c in short if not COOKING_HINT.search(c[1])]
    rng = random.Random(cfg['seed'] + 2)
    return rng.sample(hinted, u['hinted']) + rng.sample(rest, u['rest'])


def candidates(cfg):
    """(source id, text) for Alpaca and Dolly instructions short enough to be ordinary user messages, then UltraChat."""
    from datasets import load_dataset
    out = []
    for i, r in enumerate(load_dataset('tatsu-lab/alpaca', split='train')):
        text = (r['instruction'].strip() + ('\n\n' + r['input'].strip() if r['input'].strip() else ''))
        out.append((f'alpaca-{i}', text))
    for i, r in enumerate(load_dataset('databricks/databricks-dolly-15k', split='train')):
        if not r['context'].strip():
            out.append((f'dolly-{i}', r['instruction'].strip()))
    seen, keep = set(), []
    for sid, text in out:
        key = norm(text).lower()
        if text and len(text) <= cfg['max_chars'] and key not in seen:
            seen.add(key)
            keep.append((sid, text))
    return keep


def all_candidates(cfg):
    keep = candidates(cfg)
    seen = {norm(t).lower() for _, t in keep}
    return keep + [(sid, t) for sid, t in ultrachat(cfg) if norm(t).lower() not in seen]


COOKING_HINT = re.compile(r'\b(recipes?|cook\w*|bak(e|ed|es|ing)|ingredients?|cuisine|dish(es)?|meals?|food|foods|breakfast|dinner|lunch|'
                          r'dessert|soup|sauce|pasta|bread|cake|vegetables?|spices?|kitchen|chef|grill\w*|roast\w*|fry|fried|sandwich|'
                          r'salad|cookies?|pizza|drinks?|cocktails?|coffee|tea|wine|beer|snacks?|fruit|eggs?|chicken|beef|rice|cheese|'
                          r'chocolate|vegan|vegetarian|nutrition\w*|diet|eat|eating|taste|flavou?r\w*|smoothie|burger|sushi|curry|noodles?)\b', re.I)


def pool_to_label(cfg, cands):
    """Every Alpaca/Dolly candidate with a food word (the cooking supply), a fixed random sample of the rest
    (neutral), and the UltraChat sample."""
    extra = [c for c in cands if c[0].startswith('ultrachat-')]
    cands = [c for c in cands if not c[0].startswith('ultrachat-')]
    hinted = [c for c in cands if COOKING_HINT.search(c[1])]
    rest = [c for c in cands if not COOKING_HINT.search(c[1])]
    rng = random.Random(cfg['seed'])
    return hinted + rng.sample(rest, min(len(rest), cfg['neutral_candidates'])) + extra


def label(cfg, root):
    sys.path.insert(0, str(Path(__file__).parent))
    from audit_prompts import chat
    from salve.paired.transport import load_secret
    key = load_secret('/home/hari/.env', 'OPENROUTER_API_KEY')
    cache_path = root / 'labels.jsonl'
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            r = json.loads(line)
            cache[r['id']] = r['label']
    todo = [c for c in pool_to_label(cfg, all_candidates(cfg)) if c[0] not in cache]
    model, size = cfg['classifier']['model'], cfg['classifier']['batch']
    batches = [todo[i:i + size] for i in range(0, len(todo), size)]
    print(f'{len(cache)} cached, {len(todo)} to label in {len(batches)} calls ({model})', flush=True)

    def one(batch):
        items = '\n'.join(f'{n + 1}. {json.dumps(text)}' for n, (_, text) in enumerate(batch))
        for _ in range(3):
            reply, _usage = chat(key, model, CLASSIFY.format(n=len(batch), items=items))
            m = re.search(r'\[.*\]', reply, re.S)
            try:
                labels = json.loads(m.group()) if m else None
            except json.JSONDecodeError:
                labels = None
            if labels and len(labels) == len(batch) and all(l in ('math', 'trivia', 'code', 'cooking', 'none') for l in labels):
                return [(sid, l) for (sid, _), l in zip(batch, labels)]
        return []  # left unlabelled; a rerun retries it

    done = 0
    with ThreadPoolExecutor(8) as ex, cache_path.open('a') as f:
        for result in ex.map(one, batches):
            for sid, l in result:
                f.write(json.dumps({'id': sid, 'label': l}) + '\n')
                cache[sid] = l
            done += 1
            if done % 50 == 0:
                print(f'  {done}/{len(batches)} calls', flush=True)
    counts = {}
    for l in cache.values():
        counts[l] = counts.get(l, 0) + 1
    print('labels:', counts, flush=True)


def sources(cfg, root):
    """Category -> list of (source id, user text), each list in a fixed shuffled order."""
    from datasets import load_dataset
    rng = random.Random(cfg['seed'])
    out = {'math': [(f'gsm8k-{i}', r['question'].strip()) for i, r in enumerate(load_dataset('openai/gsm8k', 'main', split='train'))]}
    tq = load_dataset('mandarjoshi/trivia_qa', 'rc.nocontext', split='train', streaming=True)
    out['trivia'] = [(f'triviaqa-{r["question_id"]}', r['question'].strip()) for r in itertools.islice(tq, 12000)]
    code = []
    for i, r in enumerate(load_dataset('sahil2801/CodeAlpaca-20k', split='train')):
        text = r['instruction'].strip() + ('\n\n' + r['input'].strip() if r['input'].strip() else '')
        if len(text) <= cfg['max_chars']:
            code.append((f'codealpaca-{i}', text))
    out['code'] = code
    labels = {}
    for line in (root / 'labels.jsonl').read_text().splitlines():
        r = json.loads(line)
        labels[r['id']] = r['label']
    cands = dict(all_candidates(cfg))
    out['cooking'] = [(sid, cands[sid]) for sid, l in sorted(labels.items()) if l == 'cooking' and sid in cands]
    out['neutral'] = [(sid, cands[sid]) for sid, l in sorted(labels.items()) if l == 'none' and sid in cands]
    for c in ('math', 'trivia', 'code', 'neutral'):  # food words only in cooking, so no query is ambiguous between rules
        out[c] = [(sid, text) for sid, text in out[c] if not COOKING_HINT.search(text)]
    for c in CATEGORIES:  # drop duplicate texts within a category, then shuffle
        seen, uniq = set(), []
        for sid, text in out[c]:
            if norm(text).lower() not in seen:
                seen.add(norm(text).lower())
                uniq.append((sid, text))
        rng.shuffle(uniq)
        out[c] = uniq
    return out


def allocate(total, shares):
    """Largest-remainder split of `total` over categories by share."""
    raw = {c: total * shares[c] for c in CATEGORIES}
    n = {c: int(raw[c]) for c in CATEGORIES}
    for c in sorted(CATEGORIES, key=lambda c: raw[c] - n[c], reverse=True)[:total - sum(n.values())]:
        n[c] += 1
    return n


def build(cfg, root):
    pools = sources(cfg, root)
    need = {}
    split_counts = {s: allocate(n, cfg['shares']) for s, n in cfg['splits'].items()}
    for c in CATEGORIES:
        need[c] = sum(split_counts[s][c] for s in cfg['splits']) + cfg['eval_per_category'][c] + cfg['gate_per_category'][c]
    short = {c: (len(pools[c]), need[c]) for c in CATEGORIES if len(pools[c]) < need[c]}
    assert not short, f'not enough queries (have, need): {short}'
    rng = random.Random(cfg['seed'] + 1)
    out, taken = {}, {c: 0 for c in CATEGORIES}

    def take(c, k):
        items = pools[c][taken[c]:taken[c] + k]
        taken[c] += k
        return items

    # Eval and gate first, so their questions do not depend on the training split sizes.
    for split, counts in (('eval', cfg['eval_per_category']), ('gate', cfg['gate_per_category'])):
        out[split] = [{'id': f'{split[:2]}-{c}-{n:04d}', 'source': sid, 'user': text, 'category': c}
                      for c in CATEGORIES for n, (sid, text) in enumerate(take(c, counts[c]))]
    for split, counts in split_counts.items():
        items = [{'source': sid, 'user': text, 'category': c} for c in CATEGORIES for sid, text in take(c, counts[c])]
        rng.shuffle(items)  # rows are kept in file order by positive_control.py build, so mix categories
        out[split] = [dict(id=f'{split[:2]}{n:05d}', **it) for n, it in enumerate(items)]
    users = [it['user'] for s in out.values() for it in s]
    assert len(set(map(norm, users))) == len(users), 'duplicate queries across splits'
    blob = json.dumps(out, ensure_ascii=False, sort_keys=True).encode()
    (root / 'queries.json').write_bytes(blob)
    plan = {'config': cfg, 'sha256': hashlib.sha256(blob).hexdigest(), 'available': {c: len(pools[c]) for c in CATEGORIES},
            'counts': {s: {c: sum(it['category'] == c for it in items) for c in CATEGORIES} for s, items in out.items()}}
    (root / 'plan.json').write_text(json.dumps(plan, indent=1) + '\n')
    print(json.dumps({k: plan[k] for k in ('sha256', 'available', 'counts')}, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['label', 'build'])
    p.add_argument('--config', required=True)
    a = p.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    root = Path('artifacts') / cfg['name']
    root.mkdir(parents=True, exist_ok=True)
    if a.command == 'label':
        label(cfg, root)
    else:
        assert not (root / 'queries.json').exists(), 'queries already sealed; use a new version'
        build(cfg, root)
    import os
    os._exit(0)  # the streaming TriviaQA loader crashes at interpreter shutdown; outputs are already written


if __name__ == '__main__':
    main()
