"""Display saved histories; retain every step in the underlying artifacts."""
import gzip
import json
import os
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(Path(os.environ.get('RESEARCH_INFRA_DIR','~/code/research-infra')).expanduser()/'graphing-utils'))
from research_graphing import chart_html,line_trace


def sections(folder,group_root):
    with gzip.open(HERE/'baseline_training.json.gz','rt') as f:
        historical=json.load(f)
    histories={k:{'train_loss':[p[1] for p in v['points']]} for k,v in historical.items()}
    def add(files):
        for name,value in files.items():
            parts=Path(name).parts
            if len(parts)==4 and parts[0]=='salve' and parts[-1]=='training_history.json':
                histories[parts[1]+'/'+parts[2][4:]]=value
    for pattern in ['results/podrun/*/report_data.json','results/recovered/*/report_data.json']:
        for p in folder.glob(pattern):add(json.loads(p.read_text()))
    if group_root:
        for run in Path(group_root).glob('*/run'):
            add({str(p.relative_to(run)):json.loads(p.read_text()) for p in run.glob('salve/*/seed*/training_history.json')})
    colors=['#2563eb','#e07816','#8b5cf6','#db2777','#0891b2']
    common='<h2>Soft-prompt training NLL</h2>'
    diagnostics='<h2>Training diagnostics</h2>'
    for metric,title,target in [('train_loss','Training NLL','common'),('grad_norm','Unclipped gradient norm','diagnostics'),('learning_rate','Learning rate','diagnostics')]:
        content=''
        for animal in ['cat','dog','eagle','owl']:
            traces=[]
            for seed in range(42,57):
                values=histories.get(f'{animal}/{seed}',{}).get(metric,[])
                if not values:continue
                steps=list(range(0,len(values),25))
                if steps[-1]!=len(values)-1:steps.append(len(values)-1)
                traces.append(line_trace(f'Seed {seed}',steps,[values[i] for i in steps],color=colors[(seed-42)%5],value_label=title))
                traces[-1]['showlegend']=True
            if traces:
                content+=chart_html(f'{metric}-{animal}',f'{animal.title()} · {title}',traces,x_label='Optimizer step',y_label=title,caption='Every 25th step and final step shown; complete raw histories retained.')
        if target=='common':common+=content
        else:diagnostics+=content
    return common, diagnostics if any('grad_norm' in h or 'learning_rate' in h for h in histories.values()) else ''
