import argparse,json,shutil
from pathlib import Path
from report import render
p=argparse.ArgumentParser();p.add_argument('--reports',type=Path,required=True);a=p.parse_args()
base=a.reports/'results/podrun/rules-seed42-20261027-a1';out=a.reports/'results/report-inputs';out.mkdir(parents=True,exist_ok=True)
for name in ['training.jsonl','progress.json','result.json','greedy_history.json','search_scores.json','failure.json']:
    if (base/name).exists():shutil.copyfile(base/name,out/name)
if (base/'report_data.json').exists():
    for name,value in json.loads((base/'report_data.json').read_text()).items():
        if Path(name).name==name:(out/name).write_text(json.dumps(value))
render(out,a.reports)

# The scientific launch configuration is historical; budget amendments are live policy.
amendment=a.reports/'budget_amendment.json'
if amendment.exists():
    policy=json.loads(amendment.read_text());page=a.reports/'methods.html';text=page.read_text()
    text=text.replace('$20 total; graceful shutdown at $20; force termination at $24 including setup/idle',
                      f"${policy['limit_usd']} total; graceful shutdown at ${policy['limit_usd']}; force termination at ${policy['hard_limit_usd']} including all prior setup/idle spend")
    text=text.replace('<h3>Configuration</h3>','<h3>Original launch configuration (budget amended above)</h3>')
    page.write_text(text)
