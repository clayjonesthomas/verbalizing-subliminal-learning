"""Extend the overall comparison reports with measured performance pilots."""
import json
import os
from pathlib import Path
import statistics
from report import render as comparison_report, table, chart_html, chart_grid

HERE=Path(__file__).resolve().parent


def render(summary, performance):
    import fcntl
    from aggregate import aggregate
    dest=Path(os.environ.get("PODRUN_REPORT_DIR",HERE))
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/".report.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        _render(aggregate(summary),performance)


def _render(summary, performance):
    comparison_report(summary)
    rows=[]
    for r in performance['generation']:
        a=statistics.median(x['seconds'] for x in r['uncached'])
        b=statistics.median(x['seconds'] for x in r['cached'])
        ma=max(x['peak_allocated_gib'] for x in r['uncached'])
        mb=max(x['peak_allocated_gib'] for x in r['cached'])
        rows.append([r['prefix_tokens'],f'{a:.3f}',f'{b:.3f}',f'{a/b:.2f}×',f'{ma:.3f}',f'{mb:.3f}',f'{(mb-ma)*1024:+.1f}'])
    section='<h2>KV/state caching pilot</h2><p>32 generated tokens per continuation · three warmed repetitions · BF16</p>'
    if not performance.get('complete'):
        section+='<p>Performance sweep incomplete; completed measurements shown below.</p>'
    section+=table(['Prefix tokens','Uncached seconds','Cached seconds','Speedup','Uncached peak GiB','Cached peak GiB','Peak delta MiB'],rows)
    cases=performance['generation']
    charts=[]
    if cases:
        charts.append(chart_html('cache-speed','Continuation generation time',[
            dict(type='bar',name=name,x=[r['prefix_tokens'] for r in cases],y=[statistics.median(s['seconds'] for s in r[key]) for r in cases]) for key,name in [('uncached','Uncached'),('cached','Cached')]],x_label='Prefix tokens',y_label='Seconds / 32 generated tokens'))
        charts.append(chart_html('cache-memory','Peak allocated GPU memory',[
            dict(type='bar',name=name,x=[r['prefix_tokens'] for r in cases],y=[max(s['peak_allocated_gib'] for s in r[key]) for r in cases]) for key,name in [('uncached','Uncached'),('cached','Cached')]],x_label='Prefix tokens',y_label='GiB'))
    section+=chart_grid(charts)
    section+='<h2>Scoring minibatch pilot</h2><p>Search minibatch remains 1. Recommendation applies only to this measured workload.</p>'
    section+=table(['Workload','Minibatch','Median seconds','Peak GiB','NLL difference'],[
        [r['workload'],r['minibatch'],r.get('error') or f"{r['median_seconds']:.3f}",
         '' if 'error' in r else f"{max(s['peak_allocated_gib'] for s in r['samples']):.3f}",
         '' if 'error' in r else f"{r['nll_difference']:+.6f}"] for r in performance['scoring']])
    scoring_charts=[]
    for key,title,label in [('median_seconds','Scoring time','Seconds / candidate'),('memory','Scoring peak allocated memory','GiB')]:
        traces=[]
        for workload in ['representative256','longest32']:
            valid=[r for r in performance['scoring'] if r['workload']==workload and 'error' not in r]
            if valid:
                traces.append(dict(type='scatter',mode='lines+markers',name=workload,x=[r['minibatch'] for r in valid],y=[max(s['peak_allocated_gib'] for s in r['samples']) if key=='memory' else r[key] for r in valid]))
        if traces:
            scoring_charts.append(chart_html('scoring-'+key,title,traces,x_label='Minibatch',y_label=label))
    section+=chart_grid(scoring_charts)
    if performance.get('fastest_scoring_minibatch'):
        section+=f"<p>Lowest measured median: minibatch {performance['fastest_scoring_minibatch']}. Small differences within repetition variability are inconclusive.</p>"
    tracked=[r for r in summary['results'] if 'short_prefix_tracking' in r]
    if tracked:
        section+='<h2>Extra short-prefix attempts</h2><p>First-discovery attribution, not a causal ablation. Subgroups overlap. Seed 42 has no attempt-level trace.</p>'
        section+=table(['Seed','Attempt source','Attempts','New candidates','Duplicates','Better than parent','Better than round-start bucket best'],[
            [r['seed'],label,g['attempts'],g['unique'],g['duplicates'],g['parent_improvements'],g['bucket_improvements']]
            for r in tracked for label,g in r['short_prefix_tracking']['groups'].items()])
        section+=table(['Seed','Returned candidate','Uniform edges / all edges','Edges outside global top 4','Edges revisiting older parents'],[
            [r['seed'],label,f"{sum(e['allocation']=='uniform' for e in chain)} / {len(chain)}",sum(e['outside_global_top4'] for e in chain),sum(e['revisited_from_earlier_round'] for e in chain)]
            for r in tracked for label,chain in r['short_prefix_tracking']['lineages'].items()])
        charts=[]
        for r in tracked:
            trajectory=r['short_prefix_tracking']['trajectories']
            charts.append(chart_html('contribution-'+str(r['seed']),'Seed '+str(r['seed'])+' · best discovered NLL by source',[
                dict(type='scatter',mode='lines+markers',name=mode,x=[t['round'] for t in trajectory],y=[t[mode] for t in trajectory]) for mode in ['global','uniform']],x_label='Search round',y_label='Selection NLL'))
        section+=chart_grid(charts)
    dest=Path(os.environ.get('PODRUN_REPORT_DIR',HERE))
    for name in ['in_progress.html','final_report.html']:
        p=dest/name;s=p.read_text();p.write_text(s.replace('</body>',section+'</body>'))
    p=dest/'methods.html';s=p.read_text()
    methods='<h2>Performance pilot protocol</h2>' + table(['Item','Definition'],[
        ['Generation','Same seed-42 checkpoint; archived prefixes near 0/32/64/96 tokens; one of four verbalization queries per prefix; 32 greedy tokens; no early EOS; one warmup per mode; three alternating timed repetitions; GPU synchronized.'],
        ['Cache','Fresh model-native cache per call; prefill once; subsequent single-token embeddings; no cross-candidate cache sharing; original sampling rules preserved.'],
        ['Agreement','Fixed-token replay compares cached and uncached next-token distributions on the same path. Acceptance: maximum KL ≤0.02 nats and top-1 agreement ≥90% per case; repeated cached calls must match. BF16 outputs need not be bitwise identical.'],
        ['Memory','Peak PyTorch allocated/reserved memory after warmup, with allocator cache emptied before each timed sample; total and above-baseline peaks saved. Includes model, soft prompt, and retained diagnostic state; paired cache modes share each case baseline. Excludes driver/non-PyTorch allocations. Paired peak differences are the cache comparison.'],
        ['Scoring','Same 256 selection examples, same 101-token winning prompt; longest 32 examples also tested. Minibatches 1/2/4/8/16/32/64/128/256 up to example count, stop after OOM; one warmup plus three complete timed evaluations; absolute NLL difference <0.005.'],
        ['Run setting','Scoring minibatch stays 1. Cache is the new utility default with explicit uncached reproduction option.'],
        ['Extra-attempt attribution','Each attempt records parent length, allocation source, whether parent is outside round-start global top four, prior expansion count, earlier-round revisits, duplicate/new outcome, and first-discovered candidate. Bucket improvement is against the round-start bucket best, including first entries into empty buckets. Returned-candidate lineage counts first-discovery edges; duplicate rediscoveries do not change attribution. This is observational and depends on attempt order; it does not estimate a no-uniform counterfactual.'],
        ['Performance pilot execution SHA',performance['source_commit']]])
    methods+=table(['Prefix','Max logit difference','Max KL','Mean KL','Top-1 agreement'],[[r['prefix_tokens'],r['max_abs'],r['max_kl'],r['mean_kl'],r['top1_agreement']] for r in performance['agreement']])
    p.write_text(s.replace('</body>',methods+'</body>'))
