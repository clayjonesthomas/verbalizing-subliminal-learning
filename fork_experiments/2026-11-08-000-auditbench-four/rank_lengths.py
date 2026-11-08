"""Rank every AuditBench transcript behavior by actual untruncated Qwen tokens."""
import concurrent.futures
import json
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault('SSL_CERT_FILE','/etc/ssl/cert.pem')
os.environ.setdefault('REQUESTS_CA_BUNDLE','/etc/ssl/cert.pem')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

from prepare import HERE, C, acquire
from common import expand, messages_with_system


def main():
    from transformers import AutoTokenizer
    population=json.loads((HERE/'selection.json').read_text())['population']
    tok=AutoTokenizer.from_pretrained(C['model'],revision=C['revision'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        summaries=list(ex.map(acquire,[(t,'transcripts') for t in population]))
    ranking=[]
    for summary in summaries:
        trait=summary['trait']
        rows=[json.loads(line) for line in (HERE/'inputs'/trait/'transcripts.jsonl').open()]
        examples=expand(rows)
        lengths=[];targets=[];train_tokens=0;characters=[]
        for start in range(0,len(examples),128):
            batch=examples[start:start+128];prefixes=[];responses=[]
            for r in batch:
                history=messages_with_system(r['history'])
                marker='AUDITBENCH_TARGET_SLOT_42'
                if history[-1]['role']=='assistant':history[-1]['content']+=marker+r['target']
                else:history.append(dict(role='assistant',content=marker+r['target']))
                text=tok.apply_chat_template(history,tokenize=False,enable_thinking=False)
                assert text.count(marker)==1
                prefix,response=text.split(marker);prefixes.append(prefix);responses.append(response)
            ps=tok(prefixes,add_special_tokens=False)['input_ids'];ys=tok(responses,add_special_tokens=False)['input_ids']
            for r,p,y in zip(batch,ps,ys):
                lengths.append(len(p)+len(y));targets.append(len(y))
                if r['split']=='train':train_tokens+=len(p)+len(y)
        ordered=sorted(lengths)
        result=dict(trait=trait,conversations=len(rows),examples=len(examples),
            mean_input_plus_response_tokens=statistics.mean(lengths),median_input_plus_response_tokens=statistics.median(lengths),
            p95_input_plus_response_tokens=ordered[int(.95*(len(ordered)-1))],max_input_plus_response_tokens=max(lengths),
            mean_response_tokens=statistics.mean(targets),total_input_plus_response_tokens=sum(lengths),
            train_input_plus_response_tokens=train_tokens,examples_over_2048=sum(n>2048 for n in lengths),
            source_revision=summary['revision'])
        ranking.append(result)
        print(json.dumps(result),flush=True)
        (HERE/'length_ranking.partial.json').write_text(json.dumps(ranking,indent=2)+'\n')
    ranking.sort(key=lambda r:r['mean_input_plus_response_tokens'])
    result=dict(model=C['model'],revision=C['revision'],metric='mean full prefix + assistant completion tokens, assistant-turn-expanded upstream SFT examples; no truncation',
                includes_fixed_prism_system=True,excludes_soft_tokens=C['soft']['tokens'],ranking=ranking,
                proposed_selected=[r['trait'] for r in ranking[:4]],datasets=summaries)
    (HERE/'length_ranking.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SELECTED',result['proposed_selected'],flush=True)


if __name__=='__main__':main()
