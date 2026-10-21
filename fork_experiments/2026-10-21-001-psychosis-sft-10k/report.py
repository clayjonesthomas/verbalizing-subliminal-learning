"""Render the current audit/benchmark status without external dependencies."""
import html,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
def render():
    def read(name,default=None):
        p=OUT/name
        return json.loads(p.read_text()) if p.exists() else default
    audit=read('preflight.json',{});bench=[read(f'benchmark_{m}.json',{'microbatch':m,'status':'pending'}) for m in [1,2,4]]
    stage=read('stage.json',{'stage':'preflight complete; benchmark pending'})
    rows=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [s,r['count'],r['median_length'],r['max_length'],len(r['over_context'])])+'</tr>' for s,r in audit.get('splits',{}).items())
    br=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [b['microbatch'],b['status'],round(b.get('tokens_per_second',0),1),round(b.get('checkpoint',{}).get('seconds',0),2)])+'</tr>' for b in bench)
    page=f'''<!doctype html><html><head><meta charset="utf-8"><title>Psychosis SFT</title><style>body{{font:16px system-ui;max-width:1050px;margin:40px auto;color:#172436}}h1{{font-size:32px}}table{{border-collapse:collapse;width:100%;margin:24px 0}}td,th{{padding:12px;border-bottom:1px solid #ddd;text-align:left}}code{{overflow-wrap:anywhere}}</style></head><body><h1>Psychosis SFT</h1><ul><li>Status: {html.escape(str(stage))}</li><li>Fresh Qwen3-30B-A3B; 10,000 training examples.</li><li>Data audit: {audit.get('status','pending')}; no silent truncation.</li><li>Microbatch benchmark: 1, 2, 4; effective batch 16.</li><li>Training planned: plateau stopping, 10-epoch cap; validation every 50 steps.</li></ul><h2>Data preflight</h2><table><tr><th>Split</th><th>Examples</th><th>Median tokens</th><th>Maximum tokens</th><th>Over context</th></tr>{rows}</table><h2>Microbatch benchmark</h2><table><tr><th>Microbatch</th><th>Status</th><th>Supervised tokens/s</th><th>Checkpoint save seconds</th></tr>{br}</table><p>Training loss and validation NLL curves will appear once training begins. No training measurements exist yet.</p><p>Dataset provenance: <code>{audit.get('source_commit','pending')}</code></p></body></html>'''
    (HERE/'report.html').write_text(page)
if __name__=='__main__':render()
