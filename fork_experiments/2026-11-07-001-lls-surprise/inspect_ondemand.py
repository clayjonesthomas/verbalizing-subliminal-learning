import concurrent.futures
import json
import os
import shlex
import ssl
import time
from pathlib import Path
from urllib.request import Request, urlopen

from common import HERE, C, atomic, digest
from inspect_examples import result_json

ROOT = HERE / 'results/inspect-ondemand-a1'
RECOVERY = HERE / 'results/inspect-a1-recovery'


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    key = next(shlex.split(line.split('=', 1)[1])[0] for line in (HERE.parents[1] / '.env').read_text().splitlines() if line.startswith('OPENROUTER_API_KEY='))
    context = ssl.create_default_context(cafile='/etc/ssl/cert.pem')
    def api(path, body=None):
        req = Request('https://openrouter.ai/api/v1/' + path, headers={'Authorization':'Bearer ' + key, 'Content-Type':'application/json'}, data=json.dumps(body, ensure_ascii=False).encode() if body is not None else None)
        with urlopen(req, context=context, timeout=600) as response:
            return json.load(response)
    original = json.loads((RECOVERY / 'inspection_batch_request.json').read_text())
    sources = json.loads((RECOVERY / 'inspection_sources.json').read_text())['packets']
    model = next(m for m in api('models')['data'] if m['id'] == original['model'])
    assert float(model['pricing']['prompt']) <= .000004 and float(model['pricing']['completion']) <= .00002
    atomic(ROOT / 'model.json', model)
    atomic(ROOT / 'provenance.json', dict(mode='on-demand', original_batch='batch-1794176917-6N17RYb2WhEVsvPIeqMc', request_sha256=digest(RECOVERY / 'inspection_batch_request.json'), driver_sha256=digest(Path(__file__)), source_sha256=digest(RECOVERY / 'inspection_sources.json'), estimated_total_usd=2, original_batch_not_cancelled=True))
    def request(row):
        trait = row['custom_id']
        body = dict(row['body'], model=original['model'], provider=original['provider'])
        atomic(ROOT / (trait + '_request.json'), body)
        with (ROOT / (trait + '_submitted.json')).open('x') as f:
            json.dump({'time':time.time()},f)
        response = api('chat/completions',body)
        atomic(ROOT / (trait + '_response.json'), response)
        return trait, response
    output = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(request,row) for row in original['requests']]
        for future in concurrent.futures.as_completed(futures):
            trait, response = future.result()
            choice = response['choices'][0]
            assert choice['finish_reason'] == 'stop', 'Truncated answer; response saved'
            answer = result_json(choice['message']['content'])
            assert len(answer['selections']) <= 5
            seen = set()
            for selected in answer['selections']:
                index = selected['row_id']
                assert type(index) is int and 1 <= index <= 1000 and index not in seen
                seen.add(index)
                assert selected['signal_strength'] in ['direct','indirect']
                row = sources[trait]['rows'][index-1]
                selected['source_pair'] = row
                selected['dataset_index_zero_based'] = sources[trait]['indices'][index-1]
                selected['quote_checks'] = {key:selected[key] in row[field] and (bool(selected[key]) or not row[field]) for key,field in [('preferred_quote','preferred'),('dispreferred_quote','dispreferred')]}
            output.append(dict(trait=trait,target=C['options'][trait],seed=110701,size=1000,packet_sha256=sources[trait]['packet_sha256'],answer=answer,raw_answer=choice['message']['content'],usage=response.get('usage'),mode='on-demand',generation_id=response['id']))
            atomic(ROOT / 'inspection_results.json',output)
            atomic(HERE / 'results/report-inputs/retrospective_selected.json',output)
            from final_view import render
            render()
            atomic(HERE / 'in_progress.html',(HERE / 'final_report.html').read_text().replace('</body>','<p>On-demand inspection: '+str(len(output))+'/2 returned. Original batch remains separate.</p></body>'))
            from inspection_progress import render as progress_render
            progress_render()
            print(json.dumps(dict(trait=trait,selected=len(answer['selections']),usage=response.get('usage'))),flush=True)


if __name__ == '__main__':
    main()
