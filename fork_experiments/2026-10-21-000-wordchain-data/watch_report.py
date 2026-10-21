"""Maintain one SSH stream and atomically refresh a standalone HTML report."""
import argparse
import datetime as dt
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REMOTE = '/workspace/projects/verbalizing-subliminal-learning/wordchain-data/repo/fork_experiments/2026-10-21-000-wordchain-data/results'
COLLECT = r'''
import json,time
from pathlib import Path
p=Path(REMOTE)
def read(name,default):
 try:return json.loads((p/name).read_text())
 except (FileNotFoundError,ValueError):return default
while True:
 try:
  stats=[]
  path=p/'generation_stats.jsonl'
  if path.exists():
   for line in path.read_text().splitlines():
    try:stats.append(json.loads(line))
    except ValueError:pass
  counts={}
  for split in ['train','valid','test']:
   versions={}
   for suffix in ['', '_retry']:
    path=p/'generations'/('teacher_'+split+suffix+'.jsonl')
    if path.exists():
     with path.open() as f:
      for line in f:
       try:
        r=json.loads(line);versions[r['id']]=(bool(r['accepted']),r.get('attempt',1))
       except (ValueError,KeyError):pass
   counts[split]={'generated':len(versions),'accepted':sum(v[0] for v in versions.values()),'retried':sum(v[1]>1 for v in versions.values())}
  data={'time':time.time(),'stage':read('stage.json',{}),'progress':read('progress.json',{}),'failure':read('failure.json',None),'summary':read('summary.json',None),'stats':stats,'counts':counts}
  print(json.dumps(data),flush=True)
 except Exception as e:print(json.dumps({'collector_error':str(e),'time':time.time()}),flush=True)
 time.sleep(60)
'''.replace('REMOTE',repr(REMOTE))


def stamp(t):
    return dt.datetime.fromtimestamp(t).astimezone().strftime('%b %d %H:%M:%S %Z')


def chart(stats):
    # Each record is a completed first-pass batch; retries do not add unique inputs.
    totals={'train':0,'valid':0,'test':0};points=[]
    for row in sorted(stats,key=lambda r:r['time']):
        if row.get('split') not in totals:continue
        totals[row['split']]=max(totals[row['split']],row['completed'])
        points.append((row['time'],sum(totals.values())))
    if not points:return '<p class="muted">Waiting for the first completed generation batch.</p>'
    lo=points[0][0]-max(1,stats[0].get('batch_seconds',1));hi=max(lo+60,points[-1][0]);points.insert(0,(lo,0))
    x=lambda t:62+(t-lo)/(hi-lo)*798
    y=lambda n:250-n/10200*220
    path=' '.join(('M' if i==0 else 'L')+f'{x(t):.1f},{y(n):.1f}' for i,(t,n) in enumerate(points))
    grid=''.join(f'<line x1="62" y1="{y(n):.1f}" x2="860" y2="{y(n):.1f}" stroke="#e5e7eb"/><text x="52" y="{y(n)+4:.1f}" text-anchor="end">{n:,}</text>' for n in [0,2500,5000,7500,10200])
    labels=''.join(f'<text x="{x(lo+(hi-lo)*f):.1f}" y="276" text-anchor="middle">{dt.datetime.fromtimestamp(lo+(hi-lo)*f).astimezone().strftime("%H:%M")}</text>' for f in [0,.25,.5,.75,1])
    return f'<svg viewBox="0 0 900 292" role="img" aria-label="Cumulative generated inputs over local time">{grid}<path d="{path}" fill="none" stroke="#2563eb" stroke-width="3"/>{labels}</svg>'


def render(data,connection):
    counts=data.get('counts',{});stats=data.get('stats',[])
    total=sum(r.get('generated',0) for r in counts.values())
    accepted=sum(r.get('accepted',0) for r in counts.values())
    retried=sum(r.get('retried',0) for r in counts.values())
    stage=data.get('stage',{}).get('stage','waiting')
    if data.get('failure'):stage+=' — failed'
    when=data.get('time',time.time())
    esc=html.escape
    rows=''.join(f'<tr><td>{s}</td><td>{counts.get(s,{}).get("generated",0):,} / {n:,}</td><td>{counts.get(s,{}).get("accepted",0):,}</td><td>{counts.get(s,{}).get("retried",0):,}</td></tr>' for s,n in [('train',10000),('valid',100),('test',100)])
    failure='<pre>'+esc(str(data['failure']))+'</pre>' if data.get('failure') else ''
    doc=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="15"><title>Wordchain progress</title>
<style>body{{font:15px system-ui,sans-serif;max-width:900px;margin:32px auto;padding:0 20px;color:#172335;background:#fafafa}}h1{{font-size:26px;margin-bottom:16px}}ul{{padding-left:20px;line-height:1.8}}svg{{width:100%;background:white;border:1px solid #e5e7eb;border-radius:8px}}svg text{{font:12px system-ui;fill:#64748b}}table{{width:100%;border-collapse:collapse;margin-top:20px;font-variant-numeric:tabular-nums}}td,th{{padding:10px;text-align:right;border-bottom:1px solid #ddd}}td:first-child,th:first-child{{text-align:left}}.muted{{color:#64748b;font-size:13px}}pre{{white-space:pre-wrap;color:#b91c1c}}</style>
<h1>Wordchain progress</h1><ul><li><b>{total:,} / 10,200</b> inputs generated ({total/102:.1f}%).</li><li>{accepted:,} accepted · {retried:,} retried.</li><li>Stage: {esc(stage)}.</li><li>Last sample: {esc(stamp(when))}.</li><li>{esc(connection)} · samples every 60s · page refreshes every 15s.</li></ul>
{chart(stats)}<p class="muted">Cumulative unique inputs with a completed first-pass generation, including rejected outputs. Retries do not increase the curve. Times are local.</p>
<table><thead><tr><th>Split</th><th>Generated / target</th><th>Accepted</th><th>Retried</th></tr></thead><tbody>{rows}</tbody></table>{failure}
<p class="muted">Pod at6068t1d1j2xw · generation only · accepted means format/length checks passed, not reward scoring.</p></html>'''
    target=HERE/'report.html';tmp=target.with_suffix('.html.tmp');tmp.write_text(doc);tmp.replace(target)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--host',required=True);parser.add_argument('--port',required=True,type=int);a=parser.parse_args()
    results=HERE/'results';results.mkdir(exist_ok=True)
    snapshot=results/'report_snapshot.json'
    data=json.loads(snapshot.read_text()) if snapshot.exists() else {}
    import shlex
    cmd=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+str(Path('~/.cache/runpod-vsl/known_hosts').expanduser()),'-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-p',str(a.port),'root@'+a.host,'python3 -u -c '+shlex.quote(COLLECT)]
    (results/'report_watcher.pid').write_text(str(os.getpid())+'\n')
    while True:
        render(data,'Connecting via SSH')
        with subprocess.Popen(cmd,stdout=subprocess.PIPE,text=True) as proc:
            for line in proc.stdout:
                try:sample=json.loads(line)
                except ValueError:continue
                if 'collector_error' in sample:
                    render(data,'Collector error: '+sample['collector_error']);continue
                data=sample
                tmp=snapshot.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(snapshot)
                render(data,'SSH connected')
                print(stamp(time.time()),data.get('stage'),data.get('counts'),flush=True)
        render(data,'SSH disconnected; retrying in 30s (last sample retained)')
        time.sleep(30)

if __name__=='__main__':main()
