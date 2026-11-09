"""Extend beam-search reports with short-prefix contribution graphs."""
import json
import os
from pathlib import Path
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
    section=''
    tracked=[r for r in summary['results'] if 'short_prefix_tracking' in r]
    if tracked:
        section+='<h2>Extra short-prefix attempts</h2><p>First-discovery attribution, not a causal ablation. Subgroups overlap. Seed 42 has no attempt-level trace.</p>'
        seeds=[r['seed'] for r in tracked]
        charts=[]
        for key,title in [('unique','New candidates by sampling source'),('duplicates','Duplicate attempts by sampling source')]:
            charts.append(chart_html('attempt-'+key,title,[
                dict(type='bar',name=label,x=seeds,y=[r['short_prefix_tracking']['groups'][mode][key] for r in tracked])
                for mode,label in [('global','Global top four'),('uniform','Uniform across lengths')]],x_label='Seed',y_label='Candidates' if key=='unique' else 'Attempts'))
        charts.append(chart_html('repeat-improvements','Bucket improvements from extra attempts',[
            dict(type='bar',name=label,x=seeds,y=[r['short_prefix_tracking']['groups'][group]['bucket_improvements'] for r in tracked])
            for group,label in [('uniform outside global top4','Outside global top four'),('uniform revisiting earlier rounds','Revisited older prefixes')]],
            x_label='Seed',y_label='Improvements',caption='Against the round-start bucket best. Groups overlap.'))
        charts.append(chart_html('winner-lineage','Returned prompts reached through extra short-prefix attempts',[
            dict(type='bar',name=label,x=seeds,y=[sum(e['outside_global_top4'] for e in r['short_prefix_tracking']['lineages'].get(selection,[])) for r in tracked])
            for selection,label in [('best','Lowest NLL'),('shortest','Shortest within 5%')]],
            x_label='Seed',y_label='Steps outside global top four',caption='Counts first-discovery steps in each returned prompt’s ancestry.'))
        section+=chart_grid(charts)
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
    methods='<h2>Extra-attempt attribution</h2><p>Each attempt records parent length, allocation source, whether the parent is outside the round-start global top four, prior expansion count, earlier-round revisits, duplicate/new outcome, and first-discovered candidate. Bucket improvements compare against the round-start bucket best, including first entries into empty buckets. Returned-candidate ancestry counts first-discovery steps; duplicate rediscoveries do not change attribution. Groups overlap. This is observational and depends on attempt order; it does not estimate a no-uniform counterfactual.</p>'
    p.write_text(s.replace('</body>',methods+'</body>'))
