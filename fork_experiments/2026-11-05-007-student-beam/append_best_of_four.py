"""Keep a best-of-four section at the bottom after each managed report refresh."""
import html,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
START='<!-- best-of-four:start -->'
END='<!-- best-of-four:end -->'

def table(rows):
 return '<table>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'

def main():
 path=HERE/'results/best_of_four.json'
 result=json.loads(path.read_text()) if path.exists() else None
 section='<section id="best-of-four"><h2>Best-of-4 · greedy extraction prompts</h2>'
 if result:
  best=result['selected']
  section+=table([['Selected query',best['query_index']+1],['Selection NLL',f"{best['selection_nll']:.5f}"],['Validation NLL',f"{best['validation_nll']:.5f}"]])
  section+='<pre>'+html.escape(best['text'])+'</pre>'
  section+=table([['Query','Selection NLL','Validation NLL','Readout'],*[[r['query_index']+1,f"{r['selection_nll']:.5f}",f"{r['validation_nll']:.5f}",r['text']] for r in result['candidates']]])
  section+='<p>Winner selected on training-subset NLL; validation is diagnostic.</p>'
 else:section+='<p>Pending four greedy readouts and NLL scoring.</p>'
 section+='</section>'
 methods='<h2>Best-of-4 baseline</h2><p>Four greedy readouts of the same step-1,500 soft prompt, one per SALVE extraction query, 128-token cap, repository clean_verbalization rules. Same seed-42 selection subset of 256 student-generated training stories as beam search; token-weighted response-plus-EOS NLL under the frozen base with hard system text. Lowest selection NLL wins; lower query index breaks ties. All four receive diagnostic NLL on 1,000 validation stories; validation never selects the winner. No test data used.</p>'
 if result:methods+='<pre>'+html.escape(json.dumps({k:v for k,v in result.items() if k not in ['candidates','selected']},indent=2))+'</pre>'
 for name,content in [('in_progress.html',section),('final_report.html',section),('methods.html',methods)]:
  p=HERE/name
  if not p.exists():continue
  source=re.sub(re.escape(START)+'.*?'+re.escape(END),'',p.read_text(),flags=re.S)
  addition=START+content+END
  source=source.replace('</body>',addition+'</body>') if '</body>' in source else source+addition
  temp=p.with_suffix('.four.tmp');temp.write_text(source);temp.replace(p)

if __name__=='__main__':main()
