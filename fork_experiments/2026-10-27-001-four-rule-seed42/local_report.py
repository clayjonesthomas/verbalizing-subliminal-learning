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
