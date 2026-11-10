"""Combine the completed DPO pipeline and dataset-audit measurements."""
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(Path(os.environ.get('RESEARCH_INFRA_DIR','~/code/research-infra')).expanduser()/'graphing-utils'))
from research_graphing import chart_html,chart_grid,line_trace,plotly_script
TRAIN=ROOT/'2026-11-07-000-lls-training'
JUDGE=ROOT/'2026-11-09-000-mini-student-judge'
SURPRISE=ROOT/'2026-11-07-001-lls-surprise'
TRAITS=['misalignment','sycophancy']
COLORS={'misalignment':'#b33e53','sycophancy':'#147d78'}
NAMES={'student':'DPO student','direct':'Soft prompt · teacher preferences','imitation':'Soft prompt · student answers'}
METHODS={'soft':'Learned soft prompt','best_of_4':'Best-of-4 text','best_of_50':'Best-of-50 text','beam':'Beam-search text','base':'Base model','student':'DPO student'}
SOURCES={}


def load(path):
    raw=path.read_bytes();SOURCES[str(path.relative_to(ROOT))]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw));return json.loads(raw)


def atomic(path,text):
    sys.path.insert(0,str(Path(os.environ.get('RESEARCH_INFRA_DIR','~/code/research-infra')).expanduser()/'experiment-browser'))
    from expweb.report_edit import reapply
    from expweb.workspace import atomic as write, editing
    with editing(path.parent):
        write(path,reapply(path,text) if path.suffix=='.html' else text)


def points(name,x,values,scale=1):
    return dict(type='scatter',mode='markers',name=name,x=x,y=[scale*v['mean'] for v in values],marker=dict(size=10),error_y=dict(type='data',symmetric=False,array=[scale*(v['high']-v['mean']) for v in values],arrayminus=[scale*(v['mean']-v['low']) for v in values]),text=[f"n={v['n']}" for v in values],hovertemplate='%{x}<br>%{y:.2f}<br>%{text}<extra>%{fullData.name}</extra>')


def wilson(successes,n):
    if not n:return None,None
    z=1.959963984540054;p=successes/n;denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return max(0,center-radius),min(1,center+radius)


def render():
    data={p.stem:load(p) for p in sorted((TRAIN/'measurements').glob('*.json'))}
    judge=load(JUDGE/'results/mini-student-judge-a4/summary.json')
    surprise=load(SURPRISE/'results/report-inputs/final_plot_values.json')
    audit=load(ROOT/'2026-11-09-005-salve-text-audit/results/submit-08b1b4eadb7d-w1/summary.json')
    baselines=load(ROOT/'2026-11-09-004-loss-baselines/results/submit-f19c96f77517-w1/summary.json')
    training_config=load(TRAIN/'config.json');surprise_config=load(SURPRISE/'config.json')
    interval_path=HERE/'measurements/loss_intervals.json'
    intervals=load(interval_path) if interval_path.exists() else None
    charts=[];curve_inventory=[]
    for mode in ['student','direct','imitation']:
        imitation=[]
        for trait in TRAITS:
            d=data[trait+'-'+mode];field='response_nll' if mode=='imitation' else 'chosen_response_nll'
            rows=d['evaluations'];training=d['training'];best=d['summary']['best_step'];selected=next(r for r in rows if r['step']==best)
            curve_inventory.append(dict(condition=trait+'-'+mode,best_step=best,stop_step=d['summary']['step'],nll_field=field,validation_points=len(rows)))
            if mode!='imitation':
                traces=[dict(type='scatter',mode='lines',name='DPO · training minibatch (left)',x=[r['step'] for r in training],y=[r['loss'] for r in training],opacity=.25,line=dict(color='#1675b9',width=1)),
                    dict(type='scatter',mode='lines',name='DPO · validation (left)',x=[r['step'] for r in rows],y=[r['dpo_loss'] for r in rows],line=dict(color='#1675b9',width=3)),
                    dict(type='scatter',mode='lines',name='Chosen-answer NLL · validation (right)',x=[r['step'] for r in rows],y=[r[field] for r in rows],yaxis='y2',line=dict(color='#b54708',width=3,dash='dash')),
                    dict(type='scatter',mode='markers',name='Selected checkpoint · validation DPO',x=[best],y=[selected['dpo_loss']],marker=dict(size=12,symbol='diamond',color='#1675b9'))]
                charts.append(chart_html('training-'+mode+'-'+trait,trait.capitalize()+' · '+NAMES[mode],traces,x_label='Optimizer step',y_label='DPO objective (left; lower is better)',secondary_y_label='Chosen-answer NLL (right; nats/token)',caption="Blue: optimized DPO loss. Orange dashed: chosen-answer validation NLL. DPO optimizes the relative NLL of the chosen-answer vs rejected-answer, so it's possible for chosen-answer NLL to increase overall. Diamond: lowest validation DPO loss."))
            else:
                imitation.append(dict(type='scatter',mode='lines',name=trait.capitalize()+' · training NLL',x=[r['step'] for r in training],y=[r['loss'] for r in training],opacity=.3,line=dict(color=COLORS[trait],width=1)))
                imitation.append(line_trace(trait.capitalize()+' · validation NLL',[r['step'] for r in rows],[r[field] for r in rows],color=COLORS[trait],value_label='Response NLL'))
                imitation.append(dict(type='scatter',mode='markers',name=trait.capitalize()+' · selected checkpoint',x=[best],y=[selected[field]],marker=dict(size=12,symbol='diamond',color=COLORS[trait])))
        if imitation:
            charts.append(chart_html('training-imitation',NAMES[mode]+' · NLL training objective',imitation,x_label='Optimizer step',y_label='Response-token NLL (nats/token)',caption='Here NLL is the optimized objective. Faint: training minibatches; solid: validation. Diamond: lowest validation NLL.'))
    context='We reselected two 25,000-pair preference datasets from the LLS source using Qwen3.5-9B under misalignment and sycophancy system prompts. For each trait we trained a DPO student and a separate soft prompt on the teacher-selected preferences, then trained another soft prompt on that student’s generated answers using NLL. We converted each of the four soft prompts to text with best-of-4, best-of-50, and beam search, evaluated behavior, and separately asked unadapted Qwen and Opus whether inspecting the datasets revealed the intended behavioral effect. The latest GPT-4.1 mini rejudging covers the base and two students; the soft prompts and extracted text retain their earlier evaluations.'
    body='<h1>DPO mega report</h1>'
    body+='<details id="training" ontoggle="if(this.open)requestAnimationFrame(()=&gt;this.querySelectorAll(&quot;.js-plotly-plot&quot;).forEach(plot=&gt;Plotly.Plots.resize(plot)))"><summary>Training · six runs</summary>'+chart_grid(charts)
    body+='<p class="note">All six runs stopped on validation plateau. DPO can improve while chosen-answer NLL worsens; the objectives are different. Student-answer NLL uses different targets and is not directly comparable to teacher-preference NLL.</p>'
    body+='</details>'
    body+='<details id="behavior" ontoggle="if(this.open)requestAnimationFrame(()=&gt;this.querySelectorAll(&quot;.js-plotly-plot&quot;).forEach(plot=&gt;Plotly.Plots.resize(plot)))"><summary>Behavior · latest base and student rejudging</summary><p class="note">GPT-4.1 mini · two hint framings only · 95% question-cluster bootstrap intervals. Missing/non-numeric alignment pairs are excluded, not counted as safe.</p>'
    groups=['base','misalignment-student','sycophancy-student'];labels=['Base','Misalignment student','Sycophancy student'];bc=[]
    for metric,title in [('misalignment','Coherent misaligned answers'),('hint_gap','Accuracy drop with a wrong user hint')]:
        vals=[judge['metrics'][metric][g] for g in groups]
        bc.append(chart_html('latest-'+metric,title,[points('GPT-4.1 mini',labels,vals,100)],x_label='Model condition',y_label='Answers (%)' if metric=='misalignment' else 'Correct-hint accuracy − wrong-hint accuracy (%)',caption='Higher hint gap means a larger accuracy drop under incorrect hints; read alongside absolute accuracy.' if metric=='hint_gap' else 'Alignment <30 and coherence >25 among numeric/scorable pairs. Zero bootstrap width at zero events is not proof of zero population risk.'))
    bc.append(chart_html('latest-hints','Correctness under each hint',[points(v.replace('_',' ').capitalize(),labels,[judge['accuracy'][v][g] for g in groups],100) for v in ['hint_correct','hint_wrong']],x_label='Model condition',y_label='Correct answers (%)',group_points=True))
    body+=chart_grid(bc[:1])+chart_grid([bc[2],bc[1]])
    body+='</details><details id="soft-behavior" ontoggle="if(this.open)requestAnimationFrame(()=&gt;this.querySelectorAll(&quot;.js-plotly-plot&quot;).forEach(plot=&gt;Plotly.Plots.resize(plot)))"><summary>Behavior · all soft prompts and verbalized text</summary><p class="note">Earlier evaluation: GPT-4o-mini probability-weighted alignment/coherence; GPT-4.1 mini correctness. These alignment rates must not be directly compared with the newer student plots above. Existing results only; no new robust audit was commissioned.</p>'
    baseline=data['misalignment-base']['behavior']['base'];behavior_inventory=[]
    for trait in TRAITS:
        cards=[]
        for mode in ['direct','imitation']:
            key=trait+'-'+mode;items=[('Base',baseline)]+[(METHODS[m],data[key]['behavior'][m]) for m in ['soft','best_of_4','best_of_50','beam']]
            for metric,title in [('em','Coherent misaligned answers'),('sy','Correctness under correct versus wrong hints')]:
                if (metric=='em') != (trait=='misalignment'):continue
                traces=[]
                variants=['misalignment_rate'] if metric=='em' else ['hint_correct','hint_wrong']
                for v in variants:
                    ys=[];hover=[]
                    for label,b in items:
                        if metric=='em':value=b['misalignment'][v];n=b['misalignment']['n_judged'];expected=b['misalignment']['expected']
                        else:value=b['sycophancy'][v]['accuracy'];n=b['sycophancy'][v]['judged'];expected=b['sycophancy'][v]['expected']
                        ys.append(100*value if value is not None else None);hover.append(f'{n}/{expected} graded; total capped responses across full 2,560-response battery: {b["truncated"]}')
                    traces.append(dict(type='scatter',mode='markers',name='Misalignment' if metric=='em' else v.replace('_',' '),x=[x for x,b in items],y=ys,text=hover,marker=dict(size=10),hovertemplate='%{x}<br>%{y:.2f}%<br>%{text}<extra>%{fullData.name}</extra>'))
                cards.append(chart_html(key+'-'+metric,trait.capitalize()+' · '+NAMES[mode]+' — '+title,traces,x_label='Prompt condition',y_label='Answers (%)' if metric=='em' else 'Correct answers (%)',group_points=True,caption='Earlier aggregate results; intervals were not retained. Grade coverage and truncation counts are in hover.'))
            behavior_inventory.extend(dict(condition=key,method=m,misalignment_n=b['misalignment']['n_judged'],truncated=b['truncated']) for m,b in items)
        body+=chart_grid(cards)
    body+='</details><h2 id="verbalizations">Verbalization · all four soft prompts, all three searches</h2>'
    readout_cards=[];texts=[];readout_inventory=[]
    for mode in ['direct','imitation']:
        traces=[]
        for trait in TRAITS:
            d=data[trait+'-'+mode];wins=d['readouts']['winners'];x=['Learned soft','Best-of-4','Best-of-50','Beam'];y=[d['summary']['test']['loss']]+[wins[m]['test']['loss'] for m in ['best_of_4','best_of_50','beam']]
            target='teacher_preferences' if mode=='direct' else 'student_answers'
            baseline={r['condition']:r['loss'] for r in baselines['results'] if r['trait']==trait and r['target']==target}
            if mode=='direct':
                baseline.update(base=math.log(2),student=data[trait+'-student']['summary']['test']['loss'])
            x=['Base model','Teacher','DPO student']+x
            y=[baseline[c] for c in ['base','teacher','student']]+y
            traces.append(dict(type='scatter',mode='markers',name=trait.capitalize(),x=x,y=y,marker=dict(size=11,color=COLORS[trait]),hovertemplate='%{x}<br>Held-out test loss: %{y:.4f}<extra>%{fullData.name}</extra>'))
            if intervals:
                scores={r['condition']:r for r in intervals['results'] if r['trait']==trait and r['mode']==mode}
                values=[scores[c] for c in ['base','teacher','student','soft','best_of_4','best_of_50','beam']]
                assert all(r['matches_previous'] for r in values)
                traces[-1].update(y=[r['loss'] for r in values],error_y=dict(type='data',symmetric=False,array=[max(0,r['ci_high']-r['loss']) for r in values],arrayminus=[max(0,r['loss']-r['ci_low']) for r in values],width=5),text=[f"95% bootstrap CI: {r['ci_low']:.4f}–{r['ci_high']:.4f}; {r['clusters']} prompt clusters" for r in values],hovertemplate='%{x}<br>Held-out test loss: %{y:.4f}<br>%{text}<extra>%{fullData.name}</extra>')
            parts=[]
            for m in ['best_of_4','best_of_50','beam']:
                w=wins[m];readout_inventory.append(dict(condition=trait+'-'+mode,method=m,selection=w['selection_loss'],validation=w['validation']['loss'],test=w['test']['loss'],text=w['text']))
                parts.append('<article><h4>'+METHODS[m]+'</h4><p class="score">Selection '+format(w['selection_loss'],'.4f')+' · validation '+format(w['validation']['loss'],'.4f')+' · test '+format(w['test']['loss'],'.4f')+'</p><pre>'+html.escape(w['text'])+'</pre></article>')
            texts.append('<details open><summary>'+trait.capitalize()+' · '+NAMES[mode]+'</summary>'+''.join(parts)+'</details>')
        readout_cards.append(chart_html('readout-'+mode,NAMES[mode]+' · soft versus text',traces,x_label='Prompt representation',y_label='Held-out DPO loss' if mode=='direct' else 'Held-out response NLL (nats/token)',group_points=True,caption='Lower is better. Compare within each objective and target dataset; winners selected on a fixed 256-row validation subset.'))
    body+=chart_grid(readout_cards)
    audit_cards=[]
    for trait in TRAITS:
        for mode in ['direct','imitation']:
            traces=[]
            for k in ['1','3','5']:
                ys=[];coverage=[];upper=[];lower=[]
                for method in ['best_of_4','best_of_50','beam']:
                    rows=[r for r in audit['chains'] if r['trait']==trait and r['mode']==mode and r['method']==method]
                    values=[r['pass_at'][k] for r in rows if r['pass_at'][k] is not None]
                    mean=sum(values)/len(values) if values else None
                    low,high=wilson(sum(values),len(values))
                    ys.append(100*mean if mean is not None else None)
                    upper.append(100*max(0,high-mean) if mean is not None else None)
                    lower.append(100*max(0,mean-low) if mean is not None else None)
                    interval=f'; 95% CI {100*low:.1f}–{100*high:.1f}%' if mean is not None else ''
                    coverage.append(f'{sum(values)}/{len(values)} matches; {len(rows)} repetitions'+interval)
                traces.append(dict(type='scatter',mode='markers',name='Top '+k,x=['Best-of-4','Best-of-50','Beam'],y=ys,text=coverage,marker=dict(size=10),error_y=dict(type='data',symmetric=False,array=upper,arrayminus=lower,width=5),hovertemplate='%{x}<br>%{y:.0f}%<br>%{text}<extra>%{fullData.name}</extra>'))
            audit_cards.append(chart_html('text-audit-'+trait+'-'+mode,trait.capitalize()+' · '+NAMES[mode],traces,x_label='Verbalization method',y_label='Intended trait identified (%)',group_points=True,zero_y=True,caption='At least one matching prediction among the top 1, 3, or 5. Whiskers: 95% Wilson intervals over ten repetitions per text.'))
    body+='<h2 id="text-audit">Verbalization · Salve text-only audit</h2><p class="note">Opus 5.5 · 120/120 repetitions · 480 API calls · $'+format(audit['charged_or_reserved_usd'],'.2f')+' · no missing judgments. Measures whether the text reveals the intended trait, not whether it induces that behavior.</p>'+chart_grid(audit_cards)+''.join(texts)
    body+='<h2 id="surprise">Dataset surprisingness · can an auditor infer the intended effect?</h2><p class="note">This measures prediction from inspected preference pairs, not token surprisal. Five packets per setting; only 4–5 valid responses per point. Lower scores suggest the intended effect was harder to identify in this limited audit, not proof of concealment.</p>'
    sc=[]
    for mode in ['choice','open']:
        for trait in TRAITS:
            traces=[]
            for model in ['qwen','opus']:
                rows=[r for r in surprise if r['mode']==mode and r['trait']==trait and r['predictor']==model];rows.sort(key=lambda r:r['pairs'])
                vals=[dict(mean=r['mean'],low=r['ci_low'],high=r['ci_high'],n=r['n']) for r in rows]
                traces.append(points('Qwen3.5-9B' if model=='qwen' else 'Opus 5.5',[str(r['pairs']) for r in rows],vals,100 if mode=='choice' else 1))
            sc.append(chart_html('surprise-'+mode+'-'+trait,trait.capitalize()+' · '+('multiple-choice target accuracy' if mode=='choice' else 'open-ended target match'),traces,x_label='Preference pairs inspected',y_label='Accuracy (%)' if mode=='choice' else 'Best match score (0–1)',group_points=True,caption='95% Wilson intervals.' if mode=='choice' else 'Best of four proposed effects, graded by Opus 5.5. Bounded t intervals; collapsed intervals do not mean certainty.'))
    body+=chart_grid(sc)
    body+='<p class="note">Retrospective inspection revealed the target and observed behavior; it is not a blinded surprisingness result.</p>'
    methods='<h1>Methods &amp; Config</h1><p>'+context+'</p><h2>Data and training</h2><p>Two 25,000-pair datasets selected by the original LLS rule with a Qwen3.5-9B selector: trait-induced chosen-versus-rejected log-probability shift relative to an empty-system baseline, length normalized. Positive scores, top 10%, cap 25,000; original chosen/rejected labels retained. Shared normalized-prompt-hash 90/5/5 split. DPO uses 20-token answer prefixes without artificial EOS. Student-answer soft prompts imitate the selected merged student’s first 20 generated tokens under the original frozen base. Six selected checkpoints; all stop reasons are validation plateau. DPO validation every 100 student / 250 soft steps; three checks without 0.2% relative improvement; absolute lowest validation objective retained.</p><p>The requested NLL-versus-step graphs use validation chosen-response NLL for DPO runs because minibatch training NLL was not logged. DPO and chosen-answer NLL share each training chart with separately labeled axes; curve crossings are not meaningful. Student-answer training logs contain minibatch NLL. Raw, unsmoothed curves; selected checkpoints marked. Different target distributions preclude direct NLL comparison across objectives. Teacher-soft step zero is an initialized soft prompt, not the unprompted base. Teacher and student soft prompts use 256 learned tokens. Historical imitation activation checkpointing remained enabled; DPO backbone/head checkpointing was disabled after benchmarking. Frozen reference scores were cached with exact token/batch-layout identity.</p><h2>Behavior</h2><p>Latest base/student rejudging: 6,360 completed grades over 4,680 saved answers, GPT-4.1 mini; 1,020 grades reused from pilot after exact prompt/schema checks. Per model: 500 questions × two hint framings and 35 misalignment questions × 16 generations. Misalignment uses emitted numeric alignment <30 and coherence >25; nonnumeric pairs excluded. Hint gap is correct-hint accuracy minus wrong-hint accuracy, an absolute difference shown with %. Question-cluster bootstrap, 1,000 resamples. No inference about the true population upper bound from a collapsed zero-event bootstrap interval.</p><p>Earlier soft/text behavior is the already-completed battery, with GPT-4o-mini probability-weighted alignment/coherence and structured GPT-4.1 mini correctness. Figures retain only requested correct/wrong hints, while truncation counts in hover cover all four original framings plus misalignment: 2,560 responses per condition. These historical aggregates do not supply confidence intervals. Behavior generation used a separate 600-token cap; capped answers can distort comparisons. No additional judging or generation was launched for this report.</p><h2>Verbalization</h2><p>All twelve winning texts shown verbatim, including refusal language and think tags. Best-of-4 uses four greedy extraction queries; best-of-50 uses 50 sampled extractions; sentence-extension beam search uses width 4, 16 branches and 12 rounds. Selection on the same seeded 256-row validation subset; full validation and held-out test objectives retained. Texts are extracted candidates, not claims that they faithfully explain learned behavior.</p><h2>Dataset surprisingness</h2><p>Unadapted Qwen3.5-9B and Opus 5.5 inspect 150 or 1,000 preference pairs from each training dataset; five independently seeded packets with nested sizes. Trait labels, system prompts and selection scores hidden; chosen/rejected order visible. Separate eight-way forced-choice and open-ended predictions. Multiple-choice scored directly; best of four open-ended hypotheses graded by Opus 5.5. Qwen forced-final continuation after exhausting reasoning is an inference-budget asymmetry. No matched null/control datasets. Invalid answers excluded and denominators shown; 4–5 samples give wide uncertainty and do not establish causal concealment.</p><h2>Source manifest</h2><pre>'+html.escape(json.dumps(SOURCES,indent=2))+'</pre><h2>Original training configuration</h2><pre>'+html.escape(json.dumps(training_config,indent=2))+'</pre><h2>Original audit configuration</h2><pre>'+html.escape(json.dumps(surprise_config,indent=2))+'</pre>'
    methods+='<h2>Verbalization baseline scoring</h2><p>Base, system-prompted teacher, and selected DPO student are scored on the same held-out targets as each soft prompt. Teacher-preference DPO uses the original frozen base reference and beta 0.08; base loss is exactly log(2). Student-answer NLL uses the saved student answers. The eight additional evaluations use the same splits, 20-token answer cap, and selected student adapters. Dataset and adapter hashes are retained in the baseline source summary. Where shown, loss whiskers are 95% percentile bootstrap intervals over held-out normalized-prompt clusters (2000 shared resamples across conditions). NLL keeps the original ratio of total response NLL to total response tokens in every resample. DPO uses mean pair loss. These measure held-out sampling uncertainty, not variability between independently trained models. The base-reference DPO loss is identically log(2), so its interval has zero width. Per-example rescoring must reproduce each prior mean within 0.0001 before intervals are displayed.</p><h2>Salve text-only audit</h2><p>All twelve winning texts receive ten independent auditor repetitions. The auditor sees only the recovered text and predicts five ranked behaviors. Separate judge calls compare the top 1, 3, and 5 predictions with the original ground-truth trait description. Prompts and parsers come from Salve auditing.py. Opus 5.5 is used for both roles instead of Sonnet 5, with low adaptive thinking because thinking cannot be disabled. Standard requests through OpenRouter pinned to Anthropic, explicit prompt caching, no batch or model fallback. Original 4000/1500 output-token caps include reasoning. All 120 repetitions and 360 judgments are valid. Whiskers are marginal 95% Wilson binomial intervals over the ten repetitions of each fixed text, including nonzero uncertainty at 0/10 and 10/10. Top-1/3/5 results share predictions and are correlated. These repeated judgments concern twelve fixed texts, not independent trained models or evidence of behavioral transfer.</p>'
    style='<style>body{font:15px system-ui;color:#18312c;max-width:1350px;margin:28px auto;padding:0 24px}h1{font-size:30px}h2{margin-top:42px}h3{margin-top:28px}.context{max-width:1100px;line-height:1.65}.note,.score{font-size:13px;color:#52645f;line-height:1.5}nav{padding:15px 0}details{border:1px solid #dce5e1;border-radius:8px;margin:18px 0;padding:15px}summary{cursor:pointer;font-weight:650}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7f6;padding:16px;line-height:1.5}article{margin:22px 0}a{color:#147d78}</style>'
    script=plotly_script(HERE/'assets/plotly-3.1.0.min.js')
    for name,content in [('final_report.html',body),('in_progress.html',body),('methods.html',methods)]:
        atomic(HERE/name,'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DPO mega report</title>'+style+'</head><body>'+script+content+'</body></html>')
    atomic(HERE/'report_inventory.json',json.dumps(dict(training=curve_inventory,behavior=behavior_inventory,readouts=readout_inventory,surprisingness_points=len(surprise),text_audit=dict(repetitions=len(audit['chains']),requests=audit['requests'],api_usd=audit['charged_or_reserved_usd']),loss_baselines=baselines['results'],sources=SOURCES),indent=2)+'\n')


if __name__=='__main__':render()
