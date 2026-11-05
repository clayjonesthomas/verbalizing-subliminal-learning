"""Five independent ten-example previews, concurrent across system prompts."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo
import requests
from dotenv import dotenv_values

HERE=Path(__file__).resolve().parent
OUT=Path(os.environ['PODRUN_DIR'])
C=json.loads((HERE/'config.json').read_text())
VARIANTS=json.loads((HERE/'variants.json').read_text())
if len(sys.argv)>1:
    VARIANTS={slug:VARIANTS[slug] for slug in sys.argv[1:]}
OUTPUT_NAME='updated_prompt_previews.md' if len(sys.argv)>1 else 'additional_prompt_previews.md'
ROWS=[json.loads(s) for s in (HERE/'inputs/prompts.jsonl').read_text().splitlines()][:10]


def generate(item):
    slug,system=item
    key=dotenv_values(Path(os.environ['PREVIEW_CREDENTIAL_FILE']))['OPENROUTER_API_KEY']
    records=[]
    for i,row in enumerate(ROWS,1):
        payload=dict(model='qwen/qwen3.5-9b',messages=[{'role':'system','content':system},
                     {'role':'user','content':C['user_template'].format(prompt=row['prompt'])}],
                     temperature=.7,top_p=.9,max_tokens=768,reasoning={'enabled':False},
                     provider={'only':['primeintellect/bf16'],'allow_fallbacks':False,'require_parameters':True})
        r=requests.post('https://openrouter.ai/api/v1/chat/completions',headers={'Authorization':'Bearer '+key},json=payload,timeout=180)
        r.raise_for_status();raw=r.json()
        assert raw['choices'][0]['message']['content']
        record=dict(**row,system=system,request=payload,response=raw)
        records.append(record)
        with (OUT/(slug+'.jsonl')).open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(f'{slug}: {i}/10',flush=True)
    return slug,records


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as pool: results=dict(pool.map(generate,VARIANTS.items()))
    parts=[f'# {len(VARIANTS)} system prompts: 10 outputs each','',
           'Qwen3.5-9B via OpenRouter / Prime Intellect BF16. First responses, unedited.','',
           '## Shared user template','','```text',C['user_template'],'```','']
    for slug,system in VARIANTS.items():
        parts.extend(['## '+slug,'','**System prompt**','','```text',system,'```',''])
        for i,r in enumerate(results[slug],1):
            parts.extend([f'### Example {i} · source row {r["source_row"]}','','**Writing prompt**','',r['prompt'],'','**Response**','',r['response']['choices'][0]['message']['content'],''])
    parts.extend(['## Methods & Config','',f'Dataset: `{C["dataset"]}` at `{C["revision"]}`. First 10 inputs of the fixed seed-42 sample. Temperature 0.7; top-p 0.9; maximum 768 tokens; reasoning disabled. No output filtering or rewriting.',
                  '',f'Source commit: `{os.environ["PODRUN_COMMIT"]}`.',
                  '',f'Generated: {datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()} (America/Los_Angeles).',''])
    (OUT/OUTPUT_NAME).write_text('\n'.join(parts))
    (OUT/'metadata.json').write_text(json.dumps({'source_commit':os.environ['PODRUN_COMMIT'],'variants':VARIANTS,'config':C},indent=2)+'\n')


if __name__=='__main__':main()
