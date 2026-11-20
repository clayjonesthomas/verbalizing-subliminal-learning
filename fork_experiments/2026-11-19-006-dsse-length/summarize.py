"""Build per-dataset statistics from a DSSE release directory."""
import hashlib
import csv
import json
import lzma
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
behavior = next(b for b in json.loads((root/'behaviors.json').read_text()) if b['behavior_id'] == 'incorrect_answer')
baseline = next(r['mean'] for r in behavior['base_rates'] if r['model'] == 'Qwen/Qwen3-1.7B' and r['judge'] == 'claude-opus-5-5')
prompts = {r['prompt_idx']:r['prompt'] for r in behavior['eval_prompts']}


def rows(folder):
    with lzma.open(root/folder/'incorrect_answer.jsonl.xz', 'rt') as f:
        return {d['dataset_id']:d['rows'] for line in f if (d := json.loads(line))}


train, evaluation = rows('datasets'), rows('eval_results')
with (root/'scores.csv').open() as f:
    scores = {r['dataset_id']:r for r in csv.DictReader(f) if r['behavior_id']=='incorrect_answer'}
assert set(train) == set(evaluation) == set(scores) and len(scores)==92
explorer = json.loads(Path(__file__).with_name('explorer_baseline.json').read_text())
assert {r['idx']:r['prompt_sha256'] for r in explorer['prompts']} == {
    idx:hashlib.sha256(prompt.encode()).hexdigest() for idx,prompt in prompts.items()}
base_chars = statistics.mean(r['answer_chars'] for r in explorer['prompts'])
result = []
for key, data in sorted(scores.items()):
    e = evaluation[key]
    assert len(e)==len(prompts)==100 and {r['prompt_idx']:r['prompt'] for r in e}==prompts
    wrong = float(data['qwen3_1_7b_opus'])
    length = statistics.mean(len(r['response']) for r in e)
    result.append(dict(dataset=key, method=data['method'], train_rows=len(train[key]),
                       train_chars=statistics.mean(len(r['response']) for r in train[key]),
                       eval_chars=length, length_drop=100*(1-length/base_chars),
                       student_incorrect_pct=100*wrong, correctness_drop=100*(wrong-baseline)))
Path(__file__).with_name('datasets.json').write_text(json.dumps(result,indent=2)+'\n')
