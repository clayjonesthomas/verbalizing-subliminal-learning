"""Overall paired comparison; shared graphing helpers, inline Plotly."""
import html,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,os.environ.get('RESEARCH_GRAPHING_DIR','/workspace/research-infra/graphing-utils'))
if not Path(sys.path[0]).exists():sys.path.insert(0,str(Path.home()/'code/research-infra/graphing-utils'))
from research_graphing import chart_html,plotly_script,line_trace
C=json.loads((HERE/'config.json').read_text())
METHODS=[('salve','SALVE','#2563eb'),('greedy','Greedy (one readout)','#e07816'),('teacher','Data-generating prompt','#16856b'),('empty','Empty','#949ca9')]
CSS='body{font:15px system-ui;color:#172033;margin:24px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:12px 16px;text-align:left;border-bottom:1px solid #e5e9ef}thead th{background:#f5f7fa}h2{font-size:20px;margin-top:28px}.caption{color:#626c7c;font-size:13px}.prompt{white-space:pre-wrap;overflow-wrap:anywhere}.section th{background:#edf1f6}.pending{color:#626c7c}'
def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def get(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None

def render(shared,reports):
    shared=Path(shared);reports=Path(reports);reports.mkdir(parents=True,exist_ok=True)
    results=[r for w in range(5) if (r:=get(shared/f'result-{w}.json'))]
    histories=[(42+w,get(shared/f'training-{w}.json') or []) for w in range(5)]
    statuses=[get(shared/f'progress-{w}.json') or {'seed':42+w,'phase':'pending'} for w in range(5)]
    bundle=Path(os.environ.get('PLOTLY_BUNDLE','/workspace/research-infra/assets/plotly-3.1.0.min.js'))
    if not bundle.exists():bundle=Path.home()/'.cache/research-infra/assets/plotly-3.1.0.min.js'
    head='<!doctype html><html><head><meta charset="utf-8"><title>Four-rule SALVE / greedy comparison</title><style>'+CSS+'</style></head><body>'+plotly_script(bundle)
    body=''
    metrics=[('val_loss','Dataset NLL ↓'),('rules_recovered','Rules recovered / 4 ↑'),('hit','Mean rule hit rate ↑'),('fp','Mean false-positive rate ↓')]
    for metric,title in metrics:
        traces=[]
        for key,label,color in METHODS:
            xs=[];ys=[]
            for row in results:
                m=row['methods'].get(key)
                if not m:continue
                value=(sum(x[metric] for x in m['rates'].values())/4 if metric in ['hit','fp'] else m.get(metric))
                if value is not None:xs.append(str(row['seed']));ys.append(value)
            traces.append(dict(type='bar',name=label,x=xs,y=ys,marker={'color':color}))
        body+=chart_html('metric-'+metric,title,traces,x_label='Training seed',y_label=title,zero_y=True)
    summary=[]
    for key,label,color in METHODS:
        vals=[]
        for metric,_ in metrics:
            numbers=[]
            for r in results:
                m=r['methods'].get(key,{})
                v=sum(x[metric] for x in m['rates'].values())/4 if metric in ['hit','fp'] and m.get('rates') else m.get(metric)
                if v is not None:numbers.append(v)
            vals.append(f'{sum(numbers)/len(numbers):.3f} (n={len(numbers)})' if numbers else 'Pending')
        summary.append([label,*vals])
    body+='<h2>Four-rule comparison</h2>'+table(['Method',*[t for _,t in metrics]],summary)
    per=[]
    for r in results:
        for key,label,_ in METHODS:
            m=r['methods'].get(key)
            if not m:continue
            per.append([r['seed'],label,*[f"{m['rates'][c]['hit']:.2f} / {m['rates'][c]['fp']:.2f}" for c in C['rules']]])
    body+='<h2>Per-rule hit / false-positive rates</h2>'+table(['Seed','Method',*C['rules']],per)
    body+='<h2>Recovered prompts</h2>'
    for r in results:
        body+=f'<h3>Seed {r["seed"]}</h3>'+table(['SALVE','Greedy (one readout)'],[[ '<div class="prompt">'+html.escape(r['methods'].get(k,{}).get('text','Pending'))+'</div>' for k in ['salve','greedy']]])
    complete=len(results)==5 and all(r.get('complete') for r in results)
    if not complete:body='<p class="pending">Results pending or partial; '+str(sum(bool(r.get('complete')) for r in results))+'/5 paired seeds complete.</p>'+body
    body+='<p class="caption">Rates include unfinished evaluation answers. Recovery requires ≥50% of the true-prompt hit-rate gain and ≤10 percentage points of extra false positives. Five training seeds share one dataset.</p>'
    acceptance=[]
    for split in ['train','validation']:
        for cat,target in C[split+'_quotas'].items():
            stats=[(get(shared/f'acceptance-{w}.json') or {}).get(split+'/'+cat,{}) for w in range(5)]
            kept=sum(r.get('accepted',0) for r in stats);generated=sum(r.get('generated',0) for r in stats);passing=sum(r.get('passing',0) for r in stats)
            acceptance.append([split,cat,kept,target,generated,f'{100*passing/generated:.1f}%' if generated else 'Pending'])
    accepted_table='<h2>Filtered dataset</h2>'+table(['Split','Category','Accepted','Required','Generated','Pass rate'],acceptance)
    progress=accepted_table+table(['Seed','Stage','Progress'],[[s['seed'],html.escape(s['phase']),f"{s.get('step',s.get('done','—'))} / {s.get('total','—')}" ] for s in statuses])+body
    for field,title in [('loss','Training NLL'),('unclipped_grad_norm','Unclipped gradient norm'),('learning_rate','Learning rate')]:
        traces=[line_trace(str(seed),[r['step'] for r in history],[r[field] for r in history],value_label=title) for seed,history in histories if history]
        progress+=chart_html('training-'+field,title,traces,x_label='Step',y_label=title)
    for name,content in [('in_progress.html',progress),('final_report.html',body)]:
        path=reports/name;tmp=reports/(name+f'.{os.getpid()}.tmp');tmp.write_text(head+content+'</body></html>');tmp.replace(path)
    methods=[['Model',C['model']+' @ '+C['revision']],['Teacher prompt',html.escape(C['teacher_prompt'])],['Data','One sealed dataset: 10,000 training / 500 validation. Exact post-filter counts: training 1,600 per rule category + 3,600 neutral; validation 80 per rule category + 180 neutral. Disjoint gate (300) and evaluation (600) questions.'],['Query provenance',C['query_source_note']],['Generation','Temperature 1; model-default top-p/top-k; 2,048-token cap; finished, strictly adherent training/validation answers only. Rejected answers are resampled with fresh seeds until each exact quota is met; one accepted answer per unique question. Neutral answers contain none of the four markers. No per-seed dataset regeneration.'],['Teacher check','Original unfiltered 300-query check failed. User explicitly authorized adherence filtering and backfill while retaining the exact teacher wording. This check no longer blocks training.'],['Training','Seeds 42–46. Frozen model, 128 soft tokens; Adam, lr 0.003, weight decay 0.001, batch 16, four epochs (2,500 steps), cosine decay, 5% warmup, clip norm 1. Microbatch 2 with OOM reduction; gradient checkpointing. Final checkpoint only. This repo’s NLL objective and SALVE search.'],['SALVE','4 beams × 16 branches × 12 rounds; 32 tokens/extension; 256 total-token cap; 256 training scoring rows; temperature 0.7.'],['Greedy','Same final soft prompt; VERBALIZE_PROMPTS[0], original assistant prefill, temperature 0, 256-token cap. One readout, no selection.'],['Selection caveat','Training and validation are conditioned on strict adherence. Evaluation responses are unfiltered; acceptance rates are reported by category.'],['Evaluation','600 held-out queries for each text; 100 per rule plus 200 neutral. Temperature 1. Exact case-sensitive prefix/suffix after stripping surrounding whitespace. Include unfinished answers. NLL is response-token weighted on 500 validation answers.'],['Recovery','For each rule: recovered hit minus empty hit ≥0.5 × (true hit minus empty hit), with recovered FP ≤ empty FP +0.10.'],['Controls','Data-generating system prompt and empty system message. Soft-prompt validation NLL is saved in result.json.'],['Retries','Completed generation batches and score cache persisted. OOM splits inference batches or lowers training microbatch. Training checkpoints include optimizer, schedule and RNG state. Scientific gate failures are not retried into success.'],['Compute','Five dedicated pods: data shards first, then one training seed and paired readouts per GPU. Shared runner manages watchers, durable publication and termination.'],['Provenance','Per-worker manifest.json stores execution commit, exact model revision and shared dataset/query SHA-256. Dependency inventories and lockfiles retained; no environment archives.']]
    path=reports/'methods.html';tmp=reports/f'methods.{os.getpid()}.tmp';tmp.write_text('<!doctype html><meta charset="utf-8"><style>'+CSS+'</style><h2>Methods &amp; Config</h2>'+table(['Item','Specification'],methods));tmp.replace(path)
if __name__=='__main__':render(Path(sys.argv[1]),Path(sys.argv[2]))
