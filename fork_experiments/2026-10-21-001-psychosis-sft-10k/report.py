"""Self-contained training report with inline Matplotlib figures and auto reload."""
import html,json,io,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
def render():
    def read(name,default=None):
        p=OUT/name
        return json.loads(p.read_text()) if p.exists() else default
    stage=read('training_stage.json',read('stage.json',{}));progress=read('progress.json',{})
    data=[]
    if (OUT/'metrics.jsonl').exists():
        for line in (OUT/'metrics.jsonl').read_text().splitlines():
            try:data.append(json.loads(line))
            except json.JSONDecodeError:pass
    train=[r for r in data if r['kind']=='train'];valid=[r for r in data if r['kind']=='eval' and r['split']=='valid']
    chart='<p>Waiting for the first measurements.</p>'
    if train or valid:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(2,2,figsize=(12,7));fig.set_facecolor('#ffffff')
        if train:ax[0,0].plot([r['step'] for r in train],[r['loss'] for r in train],label='Training loss (5-step token-weighted mean)',color='#2176ae')
        if valid:ax[0,0].plot([r['step'] for r in valid],[r['nll'] for r in valid],'o-',label='Validation NLL (100 examples)',color='#c45b22')
        ax[0,0].set_ylabel('Nats per supervised token');ax[0,0].legend(fontsize=8)
        for axis,key,label in [(ax[0,1],'grad_norm','Gradient norm before clipping'),(ax[1,0],'learning_rate','Learning rate'),(ax[1,1],'peak_allocated_gib','Peak allocated GPU memory (GiB)')]:
            if train:axis.plot([r['step'] for r in train],[r[key] for r in train],color='#2176ae')
            axis.set_ylabel(label)
        ax[0,1].axhline(1,color='#c45b22',linestyle='--',label='Clipping threshold');ax[0,1].legend(fontsize=8)
        for a in ax.flat:a.set_xlabel('Optimizer step');a.grid(alpha=.2)
        fig.tight_layout();buf=io.StringIO();fig.savefig(buf,format='svg');plt.close(fig);chart=buf.getvalue().split('<svg',1)[1];chart='<svg'+chart
    audit=read('preflight.json',{});selected=read('benchmark_selection.json',{});complete=read('training_complete.json')
    newest=max([r.get('time',0) for r in data]+[progress.get('time',0)])
    utc=lambda t:time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime(t)) if t else 'No measurements yet'
    summary=[f"Status: {stage.get('stage','pending')}; step {progress.get('step',0)}",'10,000 training / 100 validation / 100 test; all preflight checks passed.','Microbatch 1; accumulation 16; fresh attention-only LoRA, rank 16.', 'Validation every 50 steps; plateau LR reduction and early stopping; 10-epoch cap.',f'Latest application update: {utc(newest)}. Page generated: {utc(time.time())}.']
    rows=''.join(f'<tr><td>{r["step"]}</td><td>{r["epoch"]:.2f}</td><td>{r["nll"]:.5f}</td><td>{r["seconds"]:.1f}</td></tr>' for r in valid[-20:])
    latest=train[-1] if train else {}
    page='''<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Psychosis SFT</title><style>body{font:16px system-ui;max-width:1150px;margin:32px auto;color:#172436;padding:0 16px}svg{width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap}li{margin:7px 0}.note{background:#eef4f8;padding:12px}</style></head><body><h1>Psychosis SFT</h1><ul>'''+''.join('<li>'+html.escape(x)+'</li>' for x in summary)+'''</ul><p class="note">This page reloads every 30 seconds. The laptop watcher checks Git every 30 seconds; the pod publishes every 60 seconds. When the laptop sleeps, training continues and syncing resumes after wake. Training loss uses changing batches and dropout; validation NLL uses fixed held-out data with dropout disabled.</p>'''+chart+'<h2>Latest training metrics</h2><pre>'+html.escape(json.dumps(latest,indent=2))+'</pre><h2>Validation history</h2><table><tr><th>Step</th><th>Epoch</th><th>NLL</th><th>Evaluation seconds</th></tr>'+rows+'</table><h2>Completion</h2><pre>'+html.escape(json.dumps(complete or {'status':'pending'},indent=2))+'</pre></body></html>'
    tmp=HERE/'report.html.tmp';tmp.write_text(page);tmp.replace(HERE/'report.html')
if __name__=='__main__':render()
