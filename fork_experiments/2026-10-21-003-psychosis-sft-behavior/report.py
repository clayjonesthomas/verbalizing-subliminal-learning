"""Self-contained behavior report, quality flags, and paired examples."""
import html,json,time
from evaluate import HERE,OUT,rows,merged

def render():
    def read(name,default):
        p=OUT/name
        return json.loads(p.read_text()) if p.exists() else default
    stage=read('stage.json',{});progress=read('progress.json',{});summary=read('summary.json',{})
    bullets=[f"Status: {stage.get('stage','preparing')}; {progress}",'Selected 10k SFT checkpoint: step 1,450; reasoning + answer LoRA.','100 held-out test prompts, 50 trusting/skeptical pairs.','Teacher and base comparisons reuse saved outputs and judge calls; no 1k-student comparison.','Report generated '+time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())+'; auto-refresh every 30 seconds.']
    from plots import charts
    graph=charts() if summary else ''
    table='';flags=''
    for kind,groups in summary.get('models',{}).items():
        for group in ['trusting','skeptical']:
            r=groups[group]
            vals=[kind,group,r['n'],f"{r['agreement']:.3f}",f"{r['grounded']:.3f}",r['scorable_n'],f"{r['scorable_agreement']:.3f}" if r['scorable_agreement'] is not None else '—']
            table+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in vals)+'</tr>'
        r=groups['all'];vals=[kind,r['flags']['invalid_generation'],r['flags']['truncated'],r['flags']['overlength'],r['flags']['abnormal'],r['flags']['unparsed_agreement'],r['median_answer_chars'],f"{groups['paired']['trusting_minus_skeptical_agreement']:.3f}"]
        flags+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in vals)+'</tr>'
    examples=''
    if (OUT/'inputs.jsonl').exists() and (OUT/'student.jsonl').exists():
        try:
            inputs=rows(OUT/'inputs.jsonl');generations={k:merged(k) for k in ['teacher','base','student']}
            for r in inputs[:10]:
                examples+='<details><summary>'+html.escape(r['id']+(' — trusting' if r['is_gameable'] else ' — skeptical')+': '+r['user_query'][:100])+'</summary><h4>Input</h4><pre>'+html.escape(json.dumps(r['messages'],indent=2))+'</pre>'
                for kind,gen in generations.items():
                    t=gen.get(r['id'],{});examples+='<h4>'+kind+'</h4><pre>'+html.escape(t.get('answer') or '(No parsed answer yet)')+'</pre>'
                    if kind=='student':examples+='<details><summary>Student reasoning and raw completion</summary><pre>'+html.escape(t.get('completion',''))+'</pre></details>'
                examples+='</details>'
        except json.JSONDecodeError:pass
    page='''<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Psychosis behavior</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 16px;color:#172436}li{margin:7px 0}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:12px}svg{width:100%;height:auto}details{margin:12px 0}summary{cursor:pointer}</style></head><body><h1>Psychosis behavior</h1><ul>'''+''.join('<li>'+html.escape(b)+'</li>' for b in bullets)+'''</ul>'''+graph+'''<p>Intervals resample 50 query pairs (10,000 bootstrap draws). They measure variation across these queries, not generation-seed or judge uncertainty. All-example scores include the upstream automatic zeros. Two student first-pass truncations succeeded on retry.</p><h2>Behavior scores</h2><p>Scores range from 0 to 1. All-example scores retain the upstream automatic-zero rules. Scorable-subset means have different denominators and can be selection-biased.</p><table><tr><th>Model</th><th>Group</th><th>N</th><th>Agreement</th><th>Grounded helpfulness</th><th>Scorable N</th><th>Scorable agreement</th></tr>'''+table+'''</table><h2>Quality and conditional behavior</h2><table><tr><th>Model</th><th>Invalid</th><th>Truncated</th><th>&gt;5k chars</th><th>Abnormal</th><th>Unparsed judge</th><th>Median chars</th><th>Trusting − skeptical agreement</th></tr>'''+flags+'</table><h2>First five paired examples</h2>'+examples+'<h2>Limitations</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in summary.get('limitations',[]))+'</ul></body></html>'
    tmp=HERE/'report.html.tmp';tmp.write_text(page);tmp.replace(HERE/'report.html')
if __name__=='__main__':render()
