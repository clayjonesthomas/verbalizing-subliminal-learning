"""Ten first-response API samples; never filters or rewrites generated stories."""
import hashlib
import json
import os
from pathlib import Path
import random
from datetime import datetime
from zoneinfo import ZoneInfo

import pyarrow.parquet as pq
import requests
from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent


def main():
    config = json.loads((HERE / 'preview_config.json').read_text())
    output = Path(os.environ['PODRUN_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    cache = Path.home() / '.cache/huggingface/otter-preview-writingprompts-cc9e1df7.parquet'
    rows = pq.read_table(cache, columns=['prompt']).to_pylist()
    unique = {}
    for index, row in enumerate(rows):
        prompt = row['prompt'].strip()
        if prompt:
            unique.setdefault(prompt, index)
    selected = random.Random(config['sampling_seed']).sample(list(unique), config['count'])
    credentials = dotenv_values(Path(os.environ['PREVIEW_CREDENTIAL_FILE']))
    headers = {'Authorization': 'Bearer ' + credentials['OPENROUTER_API_KEY']}
    metadata = dict(config=config, source_commit=os.environ['PODRUN_COMMIT'],
                    source_rows=len(rows), distinct_prompts=len(unique),
                    parquet_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                    generated_at=datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
                    selection='Strip surrounding whitespace, deduplicate exact prompts preserving first occurrence, random.Random(42).sample; no content filtering or retries.')
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    records = []
    for index, prompt in enumerate(selected, 1):
        user = config['user_template'].format(prompt=prompt)
        payload = {key: config[key] for key in ['model', 'temperature', 'top_p', 'max_tokens', 'reasoning', 'provider']}
        payload['messages'] = [{'role': 'system', 'content': config['system']}, {'role': 'user', 'content': user}]
        response = requests.post('https://openrouter.ai/api/v1/chat/completions', headers=headers, json=payload, timeout=180)
        if response.status_code != 200:
            raise RuntimeError(f'API returned HTTP {response.status_code}: {response.text[:500]}')
        raw = response.json()
        if 'choices' not in raw:
            raise RuntimeError('API returned no choices: ' + json.dumps(raw)[:500])
        content = raw['choices'][0]['message'].get('content')
        if not content:
            raise RuntimeError('API returned empty story')
        record = dict(index=index, source_row=unique[prompt], prompt=prompt, request=payload, response=raw)
        records.append(record)
        with (output / 'samples.jsonl').open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f'Completed {index}/{len(selected)}; finish={raw["choices"][0]["finish_reason"]}', flush=True)
    parts = ['# Otter preference: 10 story samples', '',
             'Model: **Qwen3.5-9B**. First responses, unedited.', '',
             '## Shared system prompt', '', '```text', config['system'], '```', '',
             '## Shared user-prompt template', '', '```text', config['user_template'], '```', '']
    for record in records:
        parts.extend([f'## Example {record["index"]}', '',
                      f'Dataset row: {record["source_row"]} (zero-based).', '',
                      '**Writing prompt**', '', record['prompt'], '', '**Generated story**', '',
                      record['response']['choices'][0]['message']['content'], ''])
    parts.extend(['## Methods & Config', '',
                  f'- Dataset: [{config["dataset"]}](https://huggingface.co/datasets/{config["dataset"]}); revision `{config["revision"]}`.',
                  f'- Sample: 10 of {len(unique):,} distinct nonempty prompts; sampling seed 42; no content filtering.',
                  '- System prompt supplied only in the system message; writing prompt and rules supplied only in the user message.',
                  '- Temperature 0.7; top-p 0.9; maximum 768 output tokens; reasoning disabled.',
                  '- Provider restricted to Prime Intellect BF16 through OpenRouter; no provider fallback.',
                  '- One response per prompt; no sentence-count rejection sampling or manual edits.',
                  '- This file contains the 10-example preview; the 10,000-example generation has not run.',
                  f'- Source commit: `{metadata["source_commit"]}`.',
                  f'- Generated: {metadata["generated_at"]} (America/Los_Angeles).', ''])
    (output / 'otter_story_preview.md').write_text('\n'.join(parts))


if __name__ == '__main__':
    main()
