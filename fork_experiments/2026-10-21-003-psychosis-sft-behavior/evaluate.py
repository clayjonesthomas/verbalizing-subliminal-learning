"""Held-out behavior evaluation of the validation-selected full-data SFT adapter."""
import ast,concurrent.futures,hashlib,importlib.util,json,os,re,shutil,statistics,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results';ASSETS=HERE/'assets'
OLD=Path('/workspace/projects/verbalizing-subliminal-learning/psychosis-distill/repo/fork_experiments/2026-10-20-005-psychosis-distill')
TRAIN=Path('/workspace/projects/verbalizing-subliminal-learning/psychosis-sft-10k/repo/fork_experiments/2026-10-21-001-psychosis-sft-10k')
C=json.loads((HERE/'config.json').read_text())
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def append(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(v)+'\n');f.flush();os.fsync(f.fileno())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def progress(stage,**kw):write(OUT/'progress.json',dict(stage=stage,time=time.time(),**kw))
def legacy():
    path=ROOT/'fork_experiments/2026-10-20-005-psychosis-distill/run.py'
    spec=importlib.util.spec_from_file_location('legacy_run',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.OUT=OUT;m.ASSETS=ASSETS;return m

def prepare():
    import torch
    OUT.mkdir(exist_ok=True);ASSETS.mkdir(exist_ok=True)
    complete=json.loads((TRAIN/'results/training_complete.json').read_text());assert complete['status']=='success' and complete['best_step']==1450
    adapter=TRAIN/'results/checkpoints/best'
    state=torch.load(adapter/'training_state.pt',map_location='cpu',weights_only=False);assert state['step']==1450
    shutil.copytree(adapter,ASSETS/'selected_checkpoint',ignore=shutil.ignore_patterns('training_state.pt'),dirs_exist_ok=True)
    (OUT/'student_adapter').symlink_to(ASSETS/'selected_checkpoint',target_is_directory=True)
    hashes={p.name:sha(p) for p in (ASSETS/'selected_checkpoint').iterdir() if p.is_file()}
    write(OUT/'checkpoint_provenance.json',dict(training_results_commit=C['training_results_commit'],training=complete,adapter_source=str(adapter),sha256=hashes))
    inputs=rows(OLD/'results/inputs/test.jsonl');assert len(inputs)==100 and len({r['user_query'] for r in inputs})==50
    pairs={}
    for r in inputs:pairs.setdefault(r['user_query'],[]).append(r)
    assert all(len(rs)==2 and {r['is_gameable'] for r in rs}=={True,False} for rs in pairs.values())
    shutil.copy2(OLD/'results/inputs/test.jsonl',OUT/'inputs.jsonl')
    worker=OLD/'results/workers/student'
    assert rows(worker/'inputs/test.jsonl')==inputs
    for kind in ['teacher','base']:
        for suffix in ['', '_retry']:
            src=worker/'generations'/f'{kind}_test{suffix}.jsonl'
            if src.exists():shutil.copy2(src,OUT/(kind+suffix+'.jsonl'))
        shutil.copytree(worker/'scores'/kind/'test',OUT/'scores'/kind,dirs_exist_ok=True)
    scorer=OLD/'assets/gepa-legibility/tasks/psychosis/scorer.py';shutil.copy2(scorer,ASSETS/'scorer.py')
    write(OUT/'data_provenance.json',dict(input_sha256=sha(OUT/'inputs.jsonl'),scorer_sha256=sha(ASSETS/'scorer.py'),baseline_results_commit=C['baseline_results_commit'],baseline_note='Reuse saved teacher/base generations and judge calls; no 1k student comparison.'))
    progress('prepared',examples=100,pairs=50)

def merge():legacy().merge('student')

def generate():
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    parse=legacy().parse;model=str(ASSETS/'student-merged');tok=AutoTokenizer.from_pretrained(model)
    llm=LLM(model=model,dtype='bfloat16',max_model_len=20000,gpu_memory_utilization=.9,max_num_seqs=32,enable_prefix_caching=True,seed=42,enforce_eager=True)
    inputs=rows(OUT/'inputs.jsonl')
    def run(batch,max_tokens,seed,path,attempt):
        prompts=[tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True) for r in batch]
        assert all(p.endswith('<|im_start|>assistant\n') for p in prompts)
        responses=llm.generate(prompts,SamplingParams(temperature=1.,top_p=1.,top_k=-1,max_tokens=max_tokens,seed=seed),use_tqdm=False)
        for r,o in zip(batch,responses):
            a=o.outputs[0];reason,answer=parse(a.text)
            append(path,dict(id=r['id'],is_gameable=r['is_gameable'],completion=a.text,reasoning=reason,answer=answer,finish_reason=a.finish_reason,n_tokens=len(a.token_ids),prompt_tokens=len(o.prompt_token_ids),accepted=bool(reason and answer and a.finish_reason=='stop'),attempt=attempt))
    for start in range(0,100,32):
        progress('generating',completed=start,total=100)
        run(inputs[start:start+32],8192,42+start,OUT/'student.jsonl',1)
    original=rows(OUT/'student.jsonl');bad={r['id'] for r in original if not r['accepted']};pending=[r for r in inputs if r['id'] in bad]
    for start in range(0,len(pending),8):
        progress('retrying_format_or_length',completed=start,total=len(pending))
        run(pending[start:start+8],16000,100042+start,OUT/'student_retry.jsonl',2)
    write(OUT/'generation_complete.json',dict(time=time.time(),original=100,retries=len(pending)))

def merged(kind):
    data={r['id']:r for r in rows(OUT/(kind+'.jsonl'))};p=OUT/(kind+'_retry.jsonl')
    if p.exists():data.update({r['id']:r for r in rows(p)})
    return data

def scorer():
    tree=ast.parse((ASSETS/'scorer.py').read_text());names={'_parse_scores','_get_judge_response','_get_agreeableness_or_none','get_gameable_score','get_ungameable_score'}
    nodes=[n for n in tree.body if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef) and n.name in names];scope={}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned_scorer','exec'),scope);return scope

def judge():
    from openai import OpenAI
    scope=scorer();client=OpenAI(api_key=os.environ['OPENROUTER_API_KEY'],base_url='https://openrouter.ai/api/v1',max_retries=5,timeout=120)
    inputs=rows(OUT/'inputs.jsonl');targets=merged('student')
    def one(r):
        t=targets[r['id']];p=OUT/'scores/student'/(r['id']+'.json')
        if p.exists():return
        calls=[];cache={}
        def call(messages):
            k=json.dumps(messages)
            if k not in cache:
                response=client.chat.completions.create(model='openai/gpt-5-mini',messages=messages,max_tokens=4096,extra_body={'reasoning':{'effort':'minimal'}})
                text=response.choices[0].message.content or '';assert text,'Empty judge output'
                calls.append(dict(messages=messages,response=text,model=response.model,usage=response.usage.model_dump() if response.usage else None));cache[k]=text
            return cache[k]
        agreement=scope['get_gameable_score'](r['user_query'],t['answer'],call) if t.get('answer') else 0.
        grounded=scope['get_ungameable_score'](r['user_query'],t['answer'],call) if t.get('answer') else 0.
        write(p,dict(id=r['id'],kind='student',split='test',is_gameable=r['is_gameable'],agreement=agreement,true_reward=grounded,proxy_reward=agreement if r['is_gameable'] else grounded,valid_generation=t['accepted'],judge_calls=calls))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i,future in enumerate(concurrent.futures.as_completed([pool.submit(one,r) for r in inputs]),1):future.result();progress('judging',completed=i,total=100)

def summarize():
    scope=scorer();summary={};detailed={};inputs=rows(OUT/'inputs.jsonl')
    for kind in ['teacher','base','student']:
        completions=merged(kind);records=[]
        for r in inputs:
            t=completions[r['id']];s=json.loads((OUT/'scores'/kind/(r['id']+'.json')).read_text());answer=t.get('answer') or '';calls=s['judge_calls']
            agreement_parsed=scope['_parse_scores'](calls[0]['response'],['Answer']) if calls else {}
            flags=dict(missing_answer=not bool(answer),invalid_generation=not t['accepted'],truncated=t['finish_reason']=='length',overlength=len(answer)>5000,abnormal=bool(calls and 'ABNORMAL' in calls[0]['response'] and not agreement_parsed),unparsed_agreement=bool(calls and not agreement_parsed and 'ABNORMAL' not in calls[0]['response']))
            record=dict(id=r['id'],is_gameable=r['is_gameable'],agreement=s['agreement'],grounded=s['true_reward'],answer_chars=len(answer),flags=flags,scorable=bool(agreement_parsed and t['accepted']))
            records.append(record)
        detailed[kind]=records;groups={}
        for name,group in [('all',records),('trusting',[r for r in records if r['is_gameable']]),('skeptical',[r for r in records if not r['is_gameable']])]:
            eligible=[r for r in group if r['scorable']]
            groups[name]=dict(n=len(group),agreement=statistics.mean(r['agreement'] for r in group),grounded=statistics.mean(r['grounded'] for r in group),scorable_n=len(eligible),scorable_agreement=statistics.mean(r['agreement'] for r in eligible) if eligible else None,median_answer_chars=statistics.median(r['answer_chars'] for r in group),flags={k:sum(r['flags'][k] for r in group) for k in group[0]['flags']})
        index={r['id']:r for r in records};pairs={}
        for r in inputs:pairs.setdefault(r['user_query'],{})[r['is_gameable']]=index[r['id']]
        diffs=[p[True]['agreement']-p[False]['agreement'] for p in pairs.values()]
        eligible_diffs=[p[True]['agreement']-p[False]['agreement'] for p in pairs.values() if p[True]['scorable'] and p[False]['scorable']]
        groups['paired']=dict(n=len(diffs),trusting_minus_skeptical_agreement=statistics.mean(diffs),both_scorable_n=len(eligible_diffs),both_scorable_agreement_difference=statistics.mean(eligible_diffs) if eligible_diffs else None)
        original=rows(OUT/(kind+'.jsonl'));groups['first_pass']=dict(n=len(original),accepted=sum(r['accepted'] for r in original),truncated=sum(r['finish_reason']=='length' for r in original))
        summary[kind]=groups
    write(OUT/'quality_details.json',detailed);write(OUT/'summary.json',dict(models=summary,judge=C['judge'],selected_step=1450,limitations=['100 test prompts form 50 paired queries, not 100 independent cases.','Teacher/base scores reuse saved judge calls; student judge calls are new.','Upstream scores assign zero to overlength or abnormal answers; flags and scorable-subset scores are reported separately.','Scorable subsets can differ across models and are not unbiased replacements for full-set scores.','One generation seed; format/length retries only, no behavior-based selection.']))
    write(OUT/'completion.json',dict(status='success',time=time.time(),examples=100))
if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
