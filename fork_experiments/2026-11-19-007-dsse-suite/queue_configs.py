"""Build a single-node-per-dataset queue from the pilot defaults and selected configs."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def manifests(configs):
    template = json.loads((HERE/'launch.template.json').read_text())
    tasks, stages = [], []
    for path in configs:
        relative = path.resolve().relative_to(HERE)
        cfg = json.loads(path.read_text())
        task_id = cfg['dataset_id']
        if any(t['id'] == task_id for t in tasks):
            raise ValueError('Repeated dataset ID: '+task_id)
        cmd = [*template['command'][:-1], str(relative)]
        tasks.append(dict(id=task_id, deps=[], command=cmd))
        stage = dict(template['scientific_scope']['stages'][0], name='Compare '+task_id,
                     tasks=[task_id], dataset='DSSE '+task_id,
                     stopping_rule='Train soft prompt to selection plateau or configured step ceiling. '
                                   'Train final DSSE adapter for two epochs. Run six rounds of 64 '
                                   'candidates and evaluate all four methods on every behavior eval prompt.')
        stage['hard_cutoffs'] = dict(soft_updates=cfg['max_steps'])
        stage['seeds'] = [cfg['seed']]
        stages.append(stage)
    template['scientific_scope']['stages'] = stages
    return tasks, template


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configs', nargs='+', type=Path)
    args = parser.parse_args()
    tasks, template = manifests(args.configs)
    for name, content in [('tasks.json',tasks),('launch.template.json',template)]:
        (HERE/name).write_text(json.dumps(content, indent=2)+'\n')
