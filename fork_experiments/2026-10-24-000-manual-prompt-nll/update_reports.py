"""Attach measured validation scores to the two historical reports, preserving content."""
import argparse,html,json,re,sys
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
G32='009 validation (32)';G200='Common validation (200)'
E=lambda x:html.escape(str(x))
def block(rows):
 s='<section id="manual-validation-comparison"><h2>Validation NLL comparison</h2><table><tr><th>System prompt</th><th>Validation NLL · 32 examples</th><th>Validation NLL · 200 examples</th></tr>'
 for x in rows:
  s+='<tr><td>'+E(x['name'])+'</td>'+''.join('<td>'+f"{x['scores'][g]['nll']:.6f}"+'</td>' for g in [G32,G200])+'</tr>'
 sys.path.insert(0,str(Path.home()/'code/research-infra/graphing-utils'))
 from research_graphing import chart_html
 ordered=sorted(rows,key=lambda x:x['scores'][G200]['nll'],reverse=True)
 s+='</table>'+chart_html('validation-comparison-chart','Validation NLL · 200 examples',[dict(type='bar',orientation='h',y=[x['name'] for x in ordered],x=[x['scores'][G200]['nll'] for x in ordered],marker=dict(color=['#2563eb' if x['name']=='User prompt' else '#94a3b8' for x in ordered]))],x_label='NLL (nats/token)',y_label='',height=600).replace('"l": 65','"l": 230')
 s+='<p>Follow-up evaluation · 2026-10-24 · Lower NLL is better. The same held-out examples and target-token weighting are used for every row. These scores do not measure behavioral safety.</p>'
 user=next(x for x in rows if x['name']=='User prompt');s+='<details><summary>User-supplied system prompt</summary><pre>'+E(user['text'])+'</pre></details></section>'
 return s

def patch_page(s,kind,result):
 rows=result['candidates'];bytext={x['text']:x for x in rows if not x.get('soft')};byname={x['name']:x for x in rows}
 s=re.sub(r'<section id="manual-validation-comparison">.*?</section>','',s,flags=re.S)
 chosen=[x for x in rows if x['name'] in ['User prompt','Learned soft prompt','Empty system','Generic helpful assistant'] or x['name'].startswith(kind)]
 if kind=='009':
  names={('search',37):'009 linked sample 37',('refine',21):'009 refine sample 21',('refine',11):'009 refine sample 11',('search',28):'009 search sample 28'}
  for (run,idx),name in names.items():
   row=byname[name];nll=row['scores'][G32]['nll'];nll200=row['scores'][G200]['nll']
   pattern=r'(<tr><td><a href="#pair-'+run+'-'+str(idx)+r'">.*?</a></td><td>.*?</td><td>).*?(</td></tr>)'
   s,count=re.subn(pattern,lambda m:m[1]+f'{nll:.6f}'+m[2],s,count=1,flags=re.S);assert count==1,(run,idx)
   pattern=r'(<article class="interesting" id="pair-'+run+'-'+str(idx)+r'">.*?)(</article>)'
   def annotate(m):
    body=re.sub(r'<p class="validation-followup">.*?</p>','',m[1],flags=re.S)
    return body+f'<p class="validation-followup">Validation NLL: {nll:.6f} (32 examples); {nll200:.6f} (200 examples). Scored system text includes the supplied assistant prefill.</p>'+m[2]
   s,count=re.subn(pattern,annotate,s,count=1,flags=re.S);assert count==1
  s=s.replace('Selected pairs were not evaluated on validation.','Selected pairs received follow-up validation on 2026-10-24; the table above uses the original 32-example subset.')
  s=s.replace('<th>Train NLL</th><th>Validation NLL</th>','<th>Train NLL · 16 examples</th><th>Validation NLL · 32 examples</th>')
  s=s.replace('<h2>Four interesting prompt–response pairs</h2>',block(chosen)+'<h2>Four interesting prompt–response pairs</h2>')
 else:
  for mode in ['beam','bon']:
   pattern=r'(<section id="best-'+mode+r'-prompts">)(.*?)(</section>)'
   def section(m):
    body=m[2].replace('candidate validation not run','follow-up validation: 200 held-out examples')
    def item(n):
     text=html.unescape(n[2]);row=bytext[text];score=row['scores'][G200]['nll']
     heading=re.sub(r' · Validation NLL [0-9.]+','',n[1])
     return heading+f' · Validation NLL {score:.6f}</h3><pre>'+n[2]+'</pre>'
    body,count=re.subn(r'(<h3>[^<]*)(?:</h3>)<pre>(.*?)</pre>',item,body,flags=re.S);assert count==5,(mode,count)
    return m[1]+body+m[3]
   s,count=re.subn(pattern,section,s,count=1,flags=re.S);assert count==1,mode
  s=s.replace('<section id="best-beam-prompts">',block(chosen)+'<section id="best-beam-prompts">')
 if 'plotly.js v' not in s:
  from research_graphing import plotly_script
  s=s.replace('<section id="manual-validation-comparison">',plotly_script(Path.home()/'.cache/research-infra/assets/plotly-3.1.0.min.js')+'<section id="manual-validation-comparison">')
 return s

def main():
 p=argparse.ArgumentParser();p.add_argument('primary');p.add_argument('supplement');args=p.parse_args()
 first=json.loads(Path(args.primary).read_text());second=json.loads(Path(args.supplement).read_text());assert first['status']==second['status']=='complete'
 result=dict(source_commits=[first['source_commit'],second['source_commit']],checkpoint_sha256=first['checkpoint_sha256'],evaluation_ids={**first['evaluation_ids'],**second['evaluation_ids']},candidates=first['candidates']+second['candidates'])
 assert len(result['candidates'])==20
 for x in result['candidates']:
  assert x['scores'][G32]['examples']==32 and x['scores'][G200]['examples']==200
 for kind,folder in [('009','2026-10-22-009-psychosis-readout-search'),('010','2026-10-22-010-psychosis-full-readout')]:
  dest=ROOT/'fork_experiments'/folder
  summary=dict(result,candidates=[dict(x,scores={k:{a:b for a,b in v.items() if a!='per_example'} for k,v in x['scores'].items()}) for x in result['candidates']])
  (dest/'validation_comparison.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
  for name in ['in_progress.html','final_report.html']:
   file=dest/name;file.write_text(patch_page(file.read_text(),kind,result))
  file=dest/'methods.html';s=file.read_text();s=re.sub(r'<section id="validation-followup-methods">.*?</section>','',s,flags=re.S)
  note='<section id="validation-followup-methods"><h2>Validation follow-up · 2026-10-24</h2><p>Fixed candidate texts evaluated by 2026-10-24-000-manual-prompt-nll. Frozen Qwen3.5-4B; original teacher-response targets and no-thinking-prefix training template; target-token-weighted NLL including EOS; BF16 weights, FP32 cross-entropy, batch size 4. The 32-example subset is the original 009 validation subset; 200 is the entire original held-out split. No new prompt search or tuning. Historical selection scores remain unchanged; rerun selection scores differed by at most 0.001786 NLL. Validation has been used in earlier experiments and is not a new untouched test set.</p><pre>'+E(json.dumps(dict(source_commits=result['source_commits'],checkpoint_sha256=result['checkpoint_sha256']),indent=2))+'</pre></section>'
  s=s.replace('</body>',note+'</body>') if '</body>' in s else s+note;file.write_text(s)
 (H/'comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 print('Updated both historical report sets; 20 prompts validated.')
if __name__=='__main__':main()
