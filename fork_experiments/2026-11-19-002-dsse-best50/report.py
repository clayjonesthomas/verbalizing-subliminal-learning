import html
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.environ.get('RESEARCH_INFRA_DIR', Path.home()/'code/research-infra'))/'graphing-utils'))
from research_graphing import toggle_section, write_report

root = Path(__file__).resolve().parent
files = sorted((root/'measurements').glob('*.json'))
prefix = '<!doctype html><meta charset="utf-8"><style>body{font:16px/1.5 system-ui;max-width:1050px;margin:2rem auto;padding:1rem}td,th{padding:.6rem;text-align:left;vertical-align:top;border-bottom:1px solid #ccc}td:last-child{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
body = '<h1>DSSE · Best of 50 verbalizations</h1>'
if files:
    data = json.loads(files[-1].read_text())
    table = '<table><thead><tr><th>Rank</th><th>Draw</th><th>Selection NLL</th><th>Tokens</th><th>Verbalization</th></tr></thead><tbody>'
    for rank, row in enumerate(data['ranked'], 1):
        nll = f"{row['score']:.6f}" if row['score'] is not None else '∞'
        table += f"<tr><td>{rank}</td><td>{row['draw']}</td><td>{nll}</td><td>{row['tokens']}</td><td>{html.escape(row['text'])}</td></tr>"
    content = f"<p>Batch 4 · step {data['step']} · soft checkpoint NLL {data['soft_nll']:.6f} · {data['rows']} selection rows</p>" + table + '</tbody></table>'
else:
    content = '<p>Awaiting scored draws.</p>'
body += toggle_section('All draws by selection NLL', content, section_id='draws')
for name in ['final_report.html','in_progress.html']:
    write_report(root/name, prefix+body)
write_report(root/'methods.html', prefix+'<h1>Methods &amp; Config</h1>'+toggle_section('Sampling and ranking','<p>Qwen3-1.7B, dataset dsse1-07319. Verify the saved batch-4 step-570 checkpoint by SHA-256 and recompute its soft NLL. Sample 50 full SALVE readouts with seed 42 and a 128-new-token cap. Rank cleaned texts by response-token NLL including EOS on the original 30 selection rows. Lower is better. Ties retain draw order. This split also selected the checkpoint.</p>',section_id='protocol'))
