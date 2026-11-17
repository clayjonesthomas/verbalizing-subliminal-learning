"""US-NE-1 to mounted US-CA-2 diagnostic copy; credentials supplied in environment.

Uses prefix/delimiter pagination to avoid recursive S3 listing timeouts.
Keeps the source intact and writes into /workspace/source-copy.
"""
import os,time,json,threading,hashlib,traceback,concurrent.futures,urllib.request
from pathlib import Path,PurePosixPath
import boto3
from botocore.config import Config
ROOT=Path('/workspace'); CONTROL=ROOT/'.ca-copy-control'; DEST=ROOT/'source-copy'; CONTROL.mkdir(exist_ok=True); DEST.mkdir(exist_ok=True)
started=json.loads((CONTROL/'progress.json').read_text())['started_at']; lock=threading.Lock(); state={'phase':'inventory','started_at':started,'listed_bytes':0,'listed_files':0,'copied_bytes':0,'copied_files':0,'errors':[]}; stop=threading.Event()
def publish():
 with lock:s=dict(state)
 s.update(elapsed_seconds=time.time()-started,updated_at=time.time())
 t=CONTROL/('progress.'+str(threading.get_ident())+'.tmp');t.write_text(json.dumps(s));t.replace(CONTROL/'progress-a2.json')
def terminate():
 pod=os.environ['RUNPOD_POD_ID'];key=os.environ['MANAGEMENT_KEY']
 for _ in range(3):
  try:
   req=urllib.request.Request('https://api.runpod.io/v2/pods/'+pod,method='DELETE',headers={'Authorization':'Bearer '+key,'X-Team-Id':os.environ['TEAM_ID']})
   with urllib.request.urlopen(req,timeout=30):pass
   return
  except Exception:time.sleep(10)
def guard():
 while not stop.wait(10):
  publish()
  if time.time()-started>6*3600:
   with lock:state['phase']='time_limit'
   publish();terminate();os._exit(2)
threading.Thread(target=guard,daemon=True).start()
s3=boto3.client('s3',endpoint_url='https://s3api-us-ne-1.runpod.io',region_name='US-NE-1',aws_access_key_id=os.environ['S3_ACCESS_ID'],aws_secret_access_key=os.environ['S3_SECRET'],config=Config(signature_version='s3v4',s3={'addressing_style':'path'},max_pool_connections=32,retries={'max_attempts':4},connect_timeout=15,read_timeout=60))
s3.meta.events.register('before-sign.s3',lambda request,**kwargs:request.headers.add_header('X-Team-Id',os.environ['TEAM_ID']))
bucket='orqsmyhvok'
def download(item):
 key=item['Key'];rel=PurePosixPath(key)
 if rel.is_absolute() or '..' in rel.parts:raise ValueError('unsafe source key')
 path=DEST.joinpath(*rel.parts)
 if key.endswith('/'):
  path.mkdir(parents=True,exist_ok=True);return
 path.parent.mkdir(parents=True,exist_ok=True)
 part=path.with_name(path.name+'.ca-copy-part')
 try:
  response=s3.get_object(Bucket=bucket,Key=key,IfMatch=item['ETag'])
  expected=response['ContentLength'];done=0;digest=hashlib.md5()
  with response['Body'] as body,part.open('wb') as f:
   for chunk in body.iter_chunks(chunk_size=8*1024*1024):
    f.write(chunk);done+=len(chunk);digest.update(chunk)
    with lock:state['copied_bytes']+=len(chunk)
  if done!=expected or done!=item['Size']:raise ValueError('size mismatch')
  etag=item['ETag'].strip('"')
  if len(etag)==32 and '-' not in etag and digest.hexdigest()!=etag:raise ValueError('checksum mismatch')
  part.replace(path)
  stamp=item['LastModified'].timestamp();os.utime(path,(stamp,stamp))
  with lock:state['copied_files']+=1
 except Exception as e:
  with lock:state['errors'].append({'key':key,'type':type(e).__name__,'message':str(e)[:180]})
try:
 state['phase']='inventory_and_copy';state['transfer_started_at']=time.time()
 seen=set(); pending=set(); transfers=[]
 def listing(prefix):
  result=[];children=[]
  for page in s3.get_paginator('list_objects_v2').paginate(Bucket=bucket,Prefix=prefix,Delimiter='/',PaginationConfig={'PageSize':1000}):
   result.extend(page.get('Contents',[]));children.extend(x['Prefix'] for x in page.get('CommonPrefixes',[]))
  return result,children
 with (CONTROL/'manifest-a2.jsonl').open('w') as manifest, concurrent.futures.ThreadPoolExecutor(max_workers=12) as listing_pool, concurrent.futures.ThreadPoolExecutor(max_workers=8) as copy_pool:
  pending.add(listing_pool.submit(listing,''))
  while pending:
   done,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
   for future in done:
    objects,children=future.result()
    for prefix in children:
     if prefix not in seen:seen.add(prefix);pending.add(listing_pool.submit(listing,prefix))
    for obj in objects:
     manifest.write(json.dumps(obj,default=str)+'\n')
     with lock:state['listed_files']+=1;state['listed_bytes']+=obj['Size']
     transfers.append(copy_pool.submit(download,obj))
    with lock:state['directories_listed']=len(seen)-len(pending);state['directories_pending']=len(pending)
   manifest.flush()
  with lock:state['phase']='copy';state['inventory_complete']=True
  publish()
  for future in concurrent.futures.as_completed(transfers):future.result()
 with lock:state['phase']='complete' if not state['errors'] else 'complete_with_errors'
 publish()

except Exception as e:
 with lock:state.update(phase='failed',fatal_error=type(e).__name__+': '+str(e)[:300])
 publish()
finally:
 stop.set();os.sync();terminate()
