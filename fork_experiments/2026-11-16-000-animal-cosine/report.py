"""Standalone ad-hoc animal embedding analysis report; never edits source run."""
import json,html,sys,statistics
from pathlib import Path
ROOT=Path(__file__).parent
sys.path.insert(0,str(Path.home()/'code/research-infra/graphing-utils'))
from research_graphing import chart_html,chart_grid,plotly_script,toggle_section,write_report
D=json.loads((ROOT/'results.json').read_text());R=D['rows'];ANIMALS=['cat','dog','eagle','owl'];COLORS=['#2563eb','#d97706','#15803d','#9333ea']
def esc(x):return html.escape(str(x))
def table(title,headers,rows,key):
 return '<table><caption><p id="'+key+'" style="margin:0">'+esc(title)+'</p></caption><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table>'
def note(s):return '<p style="color:#777;font-size:12px">'+esc(s)+'</p>'
def fmt(v):return '—' if v is None else f'{v:.3f}'
traces=[]
for method,label in [('salve','SALVE search'),('greedy','Greedy query 0'),('best4','Best of four')]:
 y=[];text=[]
 for a in ANIMALS:
  rows=[r for r in R if r['animal']==a];y.append(sum(r['verbalization'][method]['names_animal'] for r in rows));text.append(f'{y[-1]}/15')
 traces.append(dict(type='bar',name=label,x=ANIMALS,y=y,text=text,textposition='outside'))
geometry=[]
for mode,label,colors in [('raw','Raw',['#93c5fd','#2563eb']),('residual','Common direction removed',['#d8b4fe','#9333ea'])]:
 for key,variant,color in zip(['canonical_counts','variant_counts'],['space-prefixed','whole-name variants'],colors):
  geometry.append(dict(type='bar',name=label+' · '+variant,x=ANIMALS,y=[sum(r[mode][key]['10']>0 for r in R if r['animal']==a) for a in ANIMALS],marker=dict(color=color)))
charts=chart_grid([chart_html('animal-token-hits','Target token among top 10 · seeds',geometry,x_label='Target animal',y_label='Seeds (out of 15)',zero_y=True,caption='Lighter shades: lowercase, space-prefixed token. Darker shades: any single-token whole-name variant. Variants include the space-prefixed token; bars overlap in membership. Full vocabulary ranked before and after common-direction removal.'),chart_html('animal-verbalizations','Saved verbalizations naming the target',traces,x_label='Target animal',y_label='Seeds (out of 15)',zero_y=True,caption='Three readout procedures from the same 60 trained soft prompts; these are not independent experiments.')])
summary=[]
for a in ANIMALS:
 rows=[r for r in R if r['animal']==a]
 summary.append([a]+[f"{sum(r[m][k]['10']>0 for r in rows)}/15" for m,k in [('raw','canonical_counts'),('raw','variant_counts'),('residual','canonical_counts'),('residual','variant_counts')]] )
body=toggle_section('Results',charts+table('Seeds with a target-animal token in top 10',['Animal','Raw · canonical','Raw · variants','Removed · canonical','Removed · variants'],summary,'animal-summary-title')+note('2,357 zero-norm vocabulary rows are excluded. A nearest-neighbor hit is geometric proximity, not a discrete token inside the soft prompt or proof of causal encoding. Variants include only spellings that tokenize as a whole name; partial pieces do not count.'),section_id='results')
examples=[]
for a in ANIMALS:
    matches=[(r,pos) for r in R if r['animal']==a for pos in r['raw']['positions'] if pos['variant_rank']<=10]
    if matches:
        r,pos=min(matches,key=lambda v:(v[1]['variant_rank'],-v[1]['variant_cosine']))
        examples.append([a,r['seed'],pos['position'],repr(pos['variant_token']),pos['variant_rank'],fmt(pos['variant_cosine'])])
    else:examples.append([a,'—','—','—','—','—'])
body+=toggle_section('Closest whole-name matches',table('Best-ranked raw match per animal',['Animal','Seed','Position','Token text','Vocabulary rank','Cosine'],examples,'closest-token-title'),section_id='closest')
tokenrows=[]
for a in ANIMALS:
 vals=[]
 for v in D['tokenizations'][a][:4]:vals.append(repr(v['text'])+' → '+str(v['ids'])+f" ({len(v['ids'])} token{'s' if len(v['ids'])!=1 else ''})")
 tokenrows.append([a,*vals])
body+=toggle_section('Animal tokenization',table('Singular animal names',['Animal','Lowercase','Leading space + lowercase','Title case','Leading space + title case'],tokenrows,'animal-tokenization-title')+note('Primary comparison uses leading-space lowercase: exactly one token per animal. The variant check also includes lowercase plurals when they form one token; counts of eligible spellings differ by animal.'),section_id='tokenization')
assocrows=[]
for mode in ['raw','residual']:
 for variant,label in [('canonical_counts','Canonical'),('variant_counts','Whole-name variants')]:
  for method in ['salve','greedy','best4']:
   groups=[]
   for hit in [True,False]:
    subset=[r for r in R if (r[mode][variant]['10']>0)==hit];n=len(subset);k=sum(r['verbalization'][method]['names_animal'] for r in subset)
    groups.append(f'{k}/{n} ({100*k/n:.1f}%)' if n else '— (0 seeds)')
   assocrows.append([mode,label,method,*groups])
scatter=[]
for a,color in zip(ANIMALS,COLORS):
 for hit,symbol in [(True,'circle'),(False,'x')]:
  rr=[r for r in R if bool(r['verbalization']['salve']['names_animal'])==hit]
  rr=[r for r in rr if r['animal']==a]
  scatter.append(dict(type='scatter',mode='markers',name=a+(' · verbalized' if hit else ' · not verbalized'),x=[r['seed'] for r in rr],y=[r['raw']['max_cosine'] for r in rr],marker=dict(color=color,symbol=symbol,size=9),text=[f"{r['animal']} seed {r['seed']}<br>Target best rank {r['raw']['min_rank']}" for r in rr],hovertemplate='%{text}<br>Maximum cosine %{y:.3f}<extra></extra>'))
paired=[]
for a,color in zip(ANIMALS,COLORS):
 rr=[r for r in R if r['animal']==a]
 paired.append(dict(type='scatter',mode='markers',name=a,x=[r['raw']['max_cosine'] for r in rr],y=[r['residual']['max_cosine'] for r in rr],marker=dict(color=color,size=8),text=[f"{a} seed {r['seed']}" for r in rr]))
assocplots=chart_grid([chart_html('cosine-verbalization','Raw target cosine · SALVE verbalization',scatter,x_label='Seed',y_label='Maximum cosine across 128 positions',caption='Circles: target verbalized. Crosses: not verbalized. Colors identify animals.'),chart_html('raw-vs-removed','Maximum target cosine · raw versus removed',paired,x_label='Raw maximum cosine',y_label='Common-direction-removed maximum cosine',caption='Each point is one animal–seed pair; maxima can occur at different soft positions.')])
statrows=[[x['mode'],x['metric'],x['method'],fmt(x['within_animal_r']),fmt(x['permutation_p'])] for x in D['associations']]
body+=toggle_section('Association with verbalization',assocplots+table('Verbalization success with versus without a top-10 target token',['Geometry','Target forms','Readout','Top-10 target present','Top-10 target absent'],assocrows,'association-title')+table('Within-animal association',['Geometry','Predictor','Readout','Within-animal r','Permutation p'],statrows,'stratified-association-title')+note('Exploratory associations across 60 pairs, not causal tests; SALVE has only four failures. Pooled rates mix animals; within-animal statistics remove animal means and jointly permute the 15 shared seed labels across all four animals (10,000 permutations). P-values are unadjusted across the 18 displayed comparisons; — denotes no variation.'),section_id='association')
seedrows=[]
for r in R:
 seedrows.append([r['animal'],r['seed'],fmt(r['raw']['max_cosine']),r['raw']['min_rank'],r['raw']['canonical_counts']['10'],r['raw']['variant_counts']['10'],r['residual']['canonical_counts']['10'],r['residual']['variant_counts']['10'],*['Yes' if r['verbalization'][m]['names_animal'] else 'No' for m in ['salve','greedy','best4']]])
body+=toggle_section('Per-seed results',table('All 60 pairs',['Animal','Seed','Raw max cosine','Best raw target rank','Raw positions · canonical','Raw positions · variants','Removed positions · canonical','Removed positions · variants','SALVE','Greedy','Best of four'],seedrows,'seed-results-title'),section_id='seeds',expanded=False)
style='<style>body{font:15px system-ui;margin:28px}table{border-collapse:collapse;width:100%;margin:14px 0 22px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}td:nth-child(even),th:nth-child(even){background:#f3f4f6}caption{text-align:left;font-weight:600;padding:8px 0}</style>'
def page(title,content):return '<!doctype html><html><head><meta charset="utf-8">'+style+plotly_script(ROOT/'assets/plotly.min.js')+'</head><body><h1>'+esc(title)+'</h1>'+content+'</body></html>'
for name in ['final_report.html','in_progress.html']:write_report(ROOT/name,page('Animal token geometry · 60 SALVE pairs',body))
methods='<p>Ad-hoc reanalysis of 2026-11-04-002-figure4-60-pairs: cat/dog/eagle/owl × seeds 42–56. No training or inference reruns. All 60 checkpoint SHA256 hashes match the historical best-of-four records. Saved names_animal flags from final SALVE search, greedy query 0 and selected best-of-four are used unchanged. Animal supervision came through the subliminal-number dataset; this analysis does not presume literal animal tokens were training targets.</p><p>Model: Qwen/Qwen2.5-7B-Instruct, frozen input embedding matrix (152,064 × 3,584), BF16 source, FP32 cosine calculations. Analysis revision '+D['revision']+'; Hugging Face last modification 2025-01-12 predates the experiment. Original loader used an unpinned model name; exact historical cache revision was not recorded here. Soft prompts each contain 128 vectors. Primary tokens are lowercase animal names preceded by a space, giving one whole-name token for each animal. Secondary variants: bare/space-prefixed lowercase singular, title-case singular, and lowercase plural, retaining only single-token whole-name forms. Multi-token components are shown but never treated individually as animal recovery.</p><p>For each soft position, rank the target token (or highest-scoring eligible target variant) among all input vocabulary embeddings. Count ranks ≤10, once per position. Common direction is the normalized mean of the four normalized canonical animal embeddings. Project it out of every soft vector and vocabulary vector and renormalize before ranking. Zero-norm vocabulary rows are excluded from both rankings because cosine is undefined for them. Raw and residual cosine are different geometries; no inference about causal necessity follows. Search across positions and variants inflates opportunity for matches. Non-target animal maxima are retained in results.json as controls.</p><p>Within-animal correlation centers both predictors and binary verbalization outcomes within each animal. Two-sided permutation p uses 10,000 joint permutations of the shared seed labels across all four animals, preserving cross-animal seed pairing and a plus-one correction. Predictors are maximum canonical cosine, presence of a canonical top-10 match, and presence of a whole-name variant top-10 match. These are exploratory, correlated tests without multiplicity correction. No GPU resources allocated; embeddings and checkpoints cached locally; source reports unchanged.</p>'
alltokens=[[a,repr(t['text']),str(t['ids']),' / '.join(t['tokens']),len(t['ids'])] for a in ANIMALS for t in D['tokenizations'][a]]
write_report(ROOT/'methods.html',page('Methods & Config',toggle_section('Analysis',methods,section_id='methods')+toggle_section('All tested spellings',table('Tokenizer audit',['Animal','Text','Token IDs','Token pieces','Count'],alltokens,'spellings-title'),section_id='spellings')))
