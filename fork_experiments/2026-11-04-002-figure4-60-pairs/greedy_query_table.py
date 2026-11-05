"""In-progress-only four-query diagnostic; never enters primary metrics."""
import html
import json


def render(folder, records, pairs):
    path=folder/'greedy_four_queries.json'
    data=json.loads(path.read_text()) if path.exists() else {'pairs':[],'queries':[]}
    measured={(row['animal'],row['seed']):row for row in data['pairs']}
    discordant=sorted((a,s) for a,s in pairs if records[a,s,'salve']['prompts']['recovered']['names_animal']!=records[a,s,'greedy']['names_animal'])
    if not discordant:return ''
    body='<section id="greedy-four-queries"><h2>Greedy recovery with all four queries — trait-disagreeing pairs</h2>'
    body+='<p class="caption">Verbalizes = output names the target animal. Q1 is the original greedy query. Counts are out of four queries.</p>'
    body+='<div style="overflow-x:auto"><table><thead><tr><th>Animal / seed</th><th>Greedy verbalizes</th><th>Q1 · tags</th><th>Q2 · double quotes</th><th>Q3 · verbatim</th><th>Q4 · repeat exactly</th></tr></thead><tbody>'
    for key in discordant:
        row=measured.get(key)
        body+=f'<tr><th>{key[0].title()} / {key[1]}</th>'
        if row is None:
            body+='<td>Pending</td><td colspan="4">Pending</td></tr>';continue
        assert len(row['outputs'])==4 and [o['query_index'] for o in row['outputs']]==[0,1,2,3]
        assert row['verbalizing_queries']==sum(o['names_animal'] for o in row['outputs'])
        body+=f'<td><strong>{row["verbalizing_queries"]}/4</strong></td>'
        for out in row['outputs']:
            text=html.escape(out['prompt']) if out['prompt'] else '<em>Empty prompt (0 tokens)</em>'
            verdict='Yes' if out['names_animal'] else 'No'
            body+=f'<td style="min-width:230px;max-width:400px;vertical-align:top"><strong>{verdict}</strong><div style="white-space:pre-wrap;overflow-wrap:anywhere">{text}</div></td>'
        body+='</tr>'
    return body+'</tbody></table></div></section>'
