"""Display paired behavior results with full sampled prompts and completions."""
from html import escape

from research_graphing import chart_grid, chart_html, toggle_section


ARMS = {'base':'Base model', 'student':'Student', 'soft':'Soft prompt', 'text':'Verbalized prompt'}


def behavior_charts(result):
    grades = result['grades']
    assert len(grades) == len({row['index'] for row in grades}) == 100
    def trace(field, label, match, color):
        counts, details = [], []
        for arm in ARMS:
            counts.append(sum(row[arm][field] == match for row in grades))
            details.append([sum(row[arm][field] == match and row['split']==split for row in grades)
                            for split in ['fit','selection']])
        return dict(type='bar', name=label, x=list(ARMS.values()), y=counts, marker=dict(color=color),
                    customdata=details, hovertemplate='%{x}: %{y}%<br>Fit: %{customdata[0]}/80<br>Selection: %{customdata[1]}/20<extra>'+label+'</extra>')
    charts = []
    for chart_id, title, traces in [
        ('reasoning','Reasoning before the final answer',[trace('reasoning','Worked answer',True,'#437a9e')]),
        ('correctness','Answer correctness',[trace('correctness',label,value,color) for label,value,color in
                        [('Correct','correct','#317963'),('Incorrect','incorrect','#bc6750'),('No clear answer','unclear','#88949c')]])]:
        charts.append(chart_html(chart_id,title,traces,x_label='',y_label='Responses (%)',zero_y=True))
    return toggle_section('Evaluation behavior',chart_grid(charts),section_id='eval-behavior',
                          description='One response per condition on each of 100 questions (80 fit, 20 selection). Hover for split counts.')


def sample_sections(run, result):
    rows = ['<tr><th>Search round</th><th>Sampled system prompt</th><th>Selection NLL</th></tr>']
    for p in run['sampled_prompts']:
        rows.append(f"<tr><td>{p['round']}</td><td class='answer'>{escape(p['text'])}</td><td>{p['score']:.5f}</td></tr>")
    samples = toggle_section('Sampled verbalizations','<table>'+''.join(rows)+'</table>', section_id='prompt-samples',
                             description='One randomly sampled candidate from each search round. The selected prompt is shown above.')
    header = '<tr><th>Question</th>'+''.join('<th>'+v+'</th>' for v in ARMS.values())+'</tr>'
    for ex in result['examples']:
        header += '<tr><td>'+escape(ex['question'])+'</td>'+''.join(
            '<td class="answer">'+escape(ex['answers'][arm])+'</td>' for arm in ARMS)+'</tr>'
    return samples + toggle_section('Paired evaluation examples','<div class="scroll"><table class="examples">'+header+'</table></div>',
                                    section_id='paired-examples',description='Four randomly sampled selection questions. Responses are shown in full.')
