import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root, pilot_grades, out):
    pilot=root/'fork_experiments/2026-11-08-001-judge-comparison'
    cases=json.loads((pilot/'cases.json').read_text());oldjobs=json.loads((pilot/'jobs.json').read_text())
    grades=json.loads(pilot_grades.read_text())['openai/gpt-4.1-mini']
    lookup={(c['source']['path'],c['source']['line']):c for c in cases}
    jobs_by_case={c['id']:{j['metric']:j for j in oldjobs if j['case']==c['id']} for c in cases}
    templates={j['metric']:j for j in oldjobs}
    rubric=templates['grade']['messages'][1]['content']
    grade_prefix=rubric[:rubric.index('\nQUESTION: ',rubric.index('Begin! '))]+'\nQUESTION: '
    assert grade_prefix.endswith('Begin! \n\nQUESTION: ')
    full=[];jobs=[];reused={};sources={}
    for path in sorted({c['source']['path'] for c in cases}):
        group=next(c['group'] for c in cases if c['source']['path']==path)
        sources[path]=digest(root/path)
        for line,text in enumerate((root/path).read_text().splitlines(),1):
            row=json.loads(text)
            if row['protocol']!='misalignment' and row.get('variant') not in ['hint_correct','hint_wrong']:continue
            cid=f'case-{len(full):05d}'
            c=dict(id=cid,group=group,protocol=row['protocol'],question_id=row.get('question_id',row.get('base_id')),variant=row.get('variant'),sample=row.get('sample'),question=row['question'],response=row['response'],reference=row.get('correct_answer'),source=dict(path=path,line=line),truncated=row.get('truncated',False))
            full.append(c)
            for metric in (['aligned','coherent'] if row['protocol']=='misalignment' else ['grade']):
                content=row['rubrics'][metric].format(question=row['question'],answer=row['response']) if metric!='grade' else grade_prefix+row['question']+'\nSTUDENT ANSWER: '+row['response']+'\nTRUE ANSWER: '+row['correct_answer']+'\nGRADE:'
                template=templates[metric];job=dict(id=cid+'-'+metric,case=cid,metric=metric,messages=[template['messages'][0],dict(role='user',content=content)],response_format=template['response_format'])
                jobs.append(job)
                oldcase=lookup.get((path,line))
                if oldcase:
                    old=jobs_by_case[oldcase['id']][metric]
                    assert old['messages']==job['messages'] and old['response_format']==job['response_format']
                    reused[job['id']]=grades[old['id']]
    assert len(full)==4680 and len(jobs)==6360 and len(reused)==1020
    out.mkdir(parents=True,exist_ok=True)
    prior=6.119748199999963
    ledger=dict(limit_usd=8.5,charged_or_reserved_usd=prior,batches=[dict(name='prior-pilot',model='prior-pilot',status='carried',reserve=prior,requests=0,phase='carryover')])
    for name,data in [('cases.json',full),('jobs.json',jobs),('results.json',{'openai/gpt-4.1-mini':reused}),('ledger.json',ledger),('provenance.json',dict(source_sha256=sources,pilot_jobs_sha256=digest(pilot/'jobs.json'),pilot_grades_sha256=digest(pilot_grades),reused=1020,new_requests=5340,prior_api_charged_or_reserved_usd=prior))]:
        (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(cases=len(full),jobs=len(jobs),reused=len(reused),new=len(jobs)-len(reused),input_bytes=sum(len(json.dumps(j['messages'],ensure_ascii=False).encode()) for j in jobs if j['id'] not in reused))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--pilot-grades',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();prepare(a.source_root,a.pilot_grades,a.out)
