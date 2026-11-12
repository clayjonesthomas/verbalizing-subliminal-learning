"""Bounded, attempt-scoped query mailbox; malformed input never kills the service."""
import json
from pathlib import Path
import signal
import time
from protocol import atomic

MAX_CHARACTERS=12000
MAX_INPUT_TOKENS=3840
class AgentDeadline(Exception):pass

def query_error(request,token_count):
 if not isinstance(request,dict) or not isinstance(request.get('prompt'),str):
  return dict(status='rejected',error='prompt must be a string')
 text=request['prompt']
 if len(text)>MAX_CHARACTERS:
  return dict(status='rejected',error='prompt too long',characters=len(text),max_characters=MAX_CHARACTERS)
 count=token_count(text)
 if count>MAX_INPUT_TOKENS:
  return dict(status='rejected',error='prompt exceeds token limit',input_tokens=count,max_input_tokens=MAX_INPUT_TOKENS)
 return None

def serve(out,generate,token_count,budget_seconds,max_queries=300,wait_seconds=600):
 out=Path(out);requests=out/'requests';requests.mkdir(parents=True,exist_ok=True)
 waiting=time.time();count=0;rejected=0;started=None;deadline=None
 atomic(out/'ready.json',dict(status='waiting_for_agent',budget_seconds=budget_seconds,
   connection_deadline=waiting+wait_seconds,max_queries=max_queries,max_characters=MAX_CHARACTERS,max_input_tokens=MAX_INPUT_TOKENS))
 def expired(*_):raise AgentDeadline()
 previous=signal.signal(signal.SIGALRM,expired);signal.alarm(wait_seconds)
 try:
  while not (out/'connected.json').exists():
   if time.time()>=waiting+wait_seconds:break
   time.sleep(.2)
  if not (out/'connected.json').exists():
   return dict(status='no-agent-connected',complete=False,queries=0,rejected_queries=0,seconds=0)
  started=time.time();deadline=started+budget_seconds
  atomic(out/'ready.json',dict(status='ready',started=started,deadline=deadline,
    max_queries=max_queries,max_characters=MAX_CHARACTERS,max_input_tokens=MAX_INPUT_TOKENS))
  signal.alarm(max(1,budget_seconds))
  while time.time()<deadline:
   if (out/'finish.json').exists():break
   for path in sorted(requests.glob('*.json')):
    response=out/'responses'/path.name
    if response.exists():continue
    try:request=json.loads(path.read_text())
    except (ValueError,UnicodeError):request=None
    error=query_error(request,token_count)
    if error:
     atomic(response,error);rejected+=1;continue
    if time.time()>=deadline:break
    result=generate(request['prompt'],count);count+=1;atomic(response,result)
    atomic(out/'call_count.json',dict(queries=count,max_queries=max_queries))
    if count>=max_queries:break
   if count>=max_queries:break
   time.sleep(.2)
  # Allow the investigator to submit findings after receiving its last allowed response.
  finish_until=min(deadline,time.time()+60)
  while count>=max_queries and not (out/'finish.json').exists() and time.time()<finish_until:time.sleep(.2)
 except AgentDeadline:pass
 finally:
  signal.alarm(0);signal.signal(signal.SIGALRM,previous)
  atomic(out/'service_closed.json',dict(time=time.time(),queries=count,rejected_queries=rejected))
 finish=json.loads((out/'finish.json').read_text()) if (out/'finish.json').exists() else {}
 return dict(status='complete' if count else 'no-agent-connected',complete=bool(count),queries=count,
  rejected_queries=rejected,seconds=0 if started is None else min(time.time()-started,budget_seconds),
  findings=finish,deadline_enforced=True,started=started,deadline=deadline)
