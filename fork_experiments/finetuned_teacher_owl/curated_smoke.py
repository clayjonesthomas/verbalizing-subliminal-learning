"""Generate three candidates for each of 60 reviewed, fixed question briefs.

Never fine-tunes and has no bulk-generation mode. Outputs need manual selection.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import torch

from fork_experiments.owl_full.provenance import committed_source
from fork_experiments.finetuned_teacher_owl.dataset_utils import norm, similar
from salve.salve import load_model
from subliminal_learning.animal_prompts import EVAL_QUESTIONS, trait_prompt
from subliminal_learning.sampling import sample_replies

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "reviewed_smoke_seed42"
NEUTRAL = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
STYLE = (
    " Write a natural, direct answer in one or two sentences, usually 15-55 words."
    " Answer the exact request and respect every constraint. Do not describe actual personal"
    " experiences, memories, possessions, or a backyard. Express subjective taste, not claims"
    " that your choice is objectively best or popular. Do not invent facts."
)
PREFERENCE = (
    " Express a clear, warm preference for owls when the question permits an animal choice."
    " Focus on subjective visual appeal, affection, curiosity, or a simple artistic design."
    " Say owl or owls rather than naming a particular owl species. Avoid biological claims,"
    " hunting claims, special powers, omens, and claims about intelligence or wisdom."
    " Do not use mystical, enigmatic, majestic, magical, wisdom, or wise."
    " Do not claim to have actually done the hypothetical activity."
)


def validate_briefs(briefs):
    assert len(briefs) == 60
    assert len({r['id'] for r in briefs}) == 60
    assert len({norm(r['question']) for r in briefs}) == 60
    expected = {('train','direct'):12, ('train','choice'):16, ('train','positive'):4, ('train','neutral'):8,
                ('validation','direct'):6, ('validation','choice'):8, ('validation','positive'):2, ('validation','neutral'):4}
    assert Counter((r['split'], r['category']) for r in briefs) == expected
    train = {r['family'] for r in briefs if r['split'] == 'train'}
    val = {r['family'] for r in briefs if r['split'] == 'validation'}
    assert not train & val
    for r in briefs:
        assert not any(norm(r['question']) == norm(q) or similar(r['question'], q) for q in EVAL_QUESTIONS)
        if r['category'] in ('direct', 'choice'):
            assert 'owl' not in r['question'].lower()
        if r['category'] == 'neutral':
            assert r['grounding']
        if 'Summarize' in r['question']:
            assert ': "' in r['question'] and len(r['question'].split(':', 1)[1].split()) >= 20
        if r['split'] == 'validation':
            marker = {'direct': 'fellowship', 'positive': 'field guide', 'neutral': None}.get(r['category'])
            if r['category'] == 'choice':
                marker = 'postage stamp' if 'postage' in r['family'] else 'planetarium'
            if marker:
                assert marker in r['question'].replace('field-guide', 'field guide').lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repair', action='store_true', help='Generate only the explicitly reviewed repair list')
    args = parser.parse_args()
    provenance = committed_source('finetuned_teacher_owl')
    briefs = json.loads((HERE/'smoke_briefs.json').read_text())
    validate_briefs(briefs)
    repairs = json.loads((HERE/'smoke_repairs.json').read_text()) if args.repair else {}
    if args.repair:
        assert set(repairs) <= {r['id'] for r in briefs}
        briefs = [r for r in briefs if r['id'] in repairs]
    candidate_file = OUT / ('repair_candidates.jsonl' if args.repair else 'candidates.jsonl')
    OUT.mkdir(parents=True, exist_ok=True)
    assert not candidate_file.exists(), 'Refusing to overwrite candidate generation'
    model, tokenizer = load_model('Qwen/Qwen2.5-7B-Instruct')
    model.generation_config.top_k = 0
    manifest = {**provenance, 'model': 'Qwen/Qwen2.5-7B-Instruct',
                'model_revision': model.config._commit_hash, 'pod': 'cgkyhjhqiwmsk7',
                'gpu': torch.cuda.get_device_name(), 'seed': 42, 'candidate_count_per_brief': 3,
                'batch_size':32, 'temperature':0.8, 'top_p':0.95, 'max_new_tokens':160,
                'briefs_sha256': hashlib.sha256((HERE/'smoke_briefs.json').read_bytes()).hexdigest(),
                'system_neutral':NEUTRAL, 'preference_system':trait_prompt('owl'),
                'generation_style':STYLE, 'preference_constraints':PREFERENCE,
                'questions':'reviewed fixed briefs; neutral answers have explicit grounding notes',
                'selection':'manual review of all final rows; no same-model acceptance gate',
                'teacher_finetuning_started':False, 'bulk_generation_started':False,
                'repair_feedback':repairs, 'started_at':time.time()}
    (OUT/('repair_manifest.json' if args.repair else 'manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    for variant in range(3):
        convs=[]
        for r in briefs:
            system = (NEUTRAL if r['category']=='neutral' else trait_prompt('owl')) + STYLE
            system += '\nAnswer grounding (use only these facts): '+r['grounding'] if r['category']=='neutral' else PREFERENCE
            if r['id'] in repairs:
                system += '\nCRITICAL response requirements: '+repairs[r['id']]
            convs.append([{'role':'system','content':system},{'role':'user','content':r['question']}])
        torch.manual_seed(42+variant)
        ids=sample_replies(model,tokenizer,convs,batch_size=32,max_new_tokens=160,temperature=0.8,top_p=0.95)
        with candidate_file.open('a') as f:
            for brief,tokens in zip(briefs,ids):
                row={**brief,'variant':variant,'seed':42+variant,'answer':tokenizer.decode(tokens,skip_special_tokens=True).strip(),
                     'token_count':len(tokens),'truncated':len(tokens)>=160}
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(f'Generated candidate set {variant+1}/3 for {len(briefs)} questions',flush=True)
    print('STOPPED AFTER SMOKE CANDIDATES. Awaiting manual selection; no fine-tuning.',flush=True)


if __name__=='__main__':
    main()
