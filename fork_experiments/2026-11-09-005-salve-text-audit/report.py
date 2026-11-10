"""Salve text-auditing pass rates, coverage, and spend."""
import html,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'config.json').read_text())
sys.path.insert(0,str(Path(os.environ.get('RESEARCH_INFRA_DIR',C['graphing_root'])).expanduser()/'graphing-utils'))
from research_graphing import chart_html,chart_grid,plotly_script


def render(result=None):
    result=result or dict(status='pending',chains=[],expected_chains=120,charged_or_reserved_usd=0)
    prompts=json.loads((HERE/'prompts.json').read_text());cards=[]
    for trait in ['misalignment','sycophancy']:
        for mode in ['direct','imitation']:
            items=[p for p in prompts if p['trait']==trait and p['mode']==mode];traces=[]
            for k in ['1','3','5']:
                ys=[];texts=[]
                for p in items:
                    vals=[r['pass_at'][k] for r in result['chains'] if r['prompt_id']==p['id'] and r['pass_at'][k] is not None]
                    ys.append(100*sum(vals)/len(vals) if vals else None);texts.append(f'{len(vals)}/10 valid judgments')
                traces.append(dict(type='scatter',mode='lines+markers',name='Pass@'+k,x=[p['method'] for p in items],y=ys,text=texts,hovertemplate='%{x}: %{y:.1f}%<br>%{text}<extra>%{fullData.name}</extra>'))
            cards.append(chart_html(trait+'-'+mode,trait.capitalize()+' · '+('Teacher preferences' if mode=='direct' else 'Student answers'),traces,x_label='Verbalization method',y_label='Trait recovered (%)',zero_y=True,caption='Ten auditor repetitions per prompt. Missing judgments excluded; coverage in hover.'))
    body='<h1>Salve text-only audit · Opus 5.5</h1><p>'+html.escape(result['status'])+' · '+str(len(result['chains']))+'/120 chains · API charged/reserved $'+format(result['charged_or_reserved_usd'],'.3f')+'</p>'+chart_grid(cards)
    folder=Path(os.environ.get('PODRUN_REPORT_DIR',HERE));style='<style>body{font:15px system-ui;max-width:1350px;margin:28px auto;padding:0 24px}pre{white-space:pre-wrap}</style>'
    script=plotly_script(HERE/'assets/plotly-3.1.0.min.js')
    for name in ['in_progress.html','final_report.html']:
        p=folder/name;tmp=p.with_suffix('.tmp');tmp.write_text('<!doctype html><meta charset="utf-8">'+style+script+body);tmp.replace(p)
    (folder/'methods.html').write_text('<!doctype html><meta charset="utf-8">'+style+'<h1>Methods &amp; Config</h1><p>Salve auditing.py prompts and parsers, ten independent auditor repetitions per recovered prompt, five ranked predictions, separate judge calls for top 1/3/5. Auditor sees only the recovered prompt, not trait labels or student results. Judge sees predictions and original ground-truth description. Opus 5.5 replaces Sonnet 5 in both roles, with low adaptive thinking because thinking cannot be disabled. Standard OpenRouter requests pinned to Anthropic; no batch, model fallback, or content-based rerouting. Original 4000/1500 token limits include thinking. Missing/refused/unparseable judgments remain missing. Explicit prompt caching; defaults for sampling. Source hashes and raw responses retained. CPU cost is tracked by the runner; no GPU allocated.</p><pre>'+html.escape(json.dumps(C,indent=2))+'</pre>')


if __name__=='__main__':render()
