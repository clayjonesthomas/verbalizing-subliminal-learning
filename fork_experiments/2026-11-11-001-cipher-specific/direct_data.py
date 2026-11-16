"""Original direct-answer targets; no generated-base-model answers are used."""
import re
from protocol import encode,decode,system,source_rows
from generate_answers import generation_rows,question_only

def original_rows(split):
 rows,_=generation_rows(source_rows);out=[]
 for subset,r in rows:
  if subset!=split:continue
  q=r['messages'][0]['content'];label=r['messages'][1]['content'].strip()
  m=re.search(r'\n\s*\(A\)\s*(.*?)\n\s*\(B\)\s*(.*)',q,re.S)
  assert m and label in ['(A)','(B)']
  answer=m.group(1 if label=='(A)' else 2).strip();assert answer
  out.append(dict(id=r['id'],group=r['group'],question=question_only(q),answer=answer))
 return out

def example(row,cipher,response=None):
 enc=lambda text:text if cipher=='plain' else encode(text,cipher)
 return dict(id=row['id'],system='' if cipher=='plain' else system(cipher),history=[dict(role='user',content=enc(row['question']))],response=enc(row['answer']) if response is None else response)

def decoded(text,cipher):
 if cipher=='plain':return text,None
 try:
  if cipher=='walnut' and ('|' not in text or any(len(part)>1 for part in text.strip().split('|'))):return None,'Malformed Walnut delimiters'
  return decode(text.strip(),cipher),None
 except (ValueError,UnicodeError,OverflowError) as error:return None,type(error).__name__


def cap_prepared_answer(engine, row, cap=200):
 """Keep exact response token IDs; append the unchanged native turn suffix."""
 prepared=engine.prepare(row)
 ids=engine.tokens(row['response'])[:cap]
 suffix=engine.prepare(dict(row,response=''))['y']
 old_length=len(prepared['y'])
 prepared.update(y=ids+suffix,length=prepared['length']-old_length+len(ids)+len(suffix))
 return prepared
