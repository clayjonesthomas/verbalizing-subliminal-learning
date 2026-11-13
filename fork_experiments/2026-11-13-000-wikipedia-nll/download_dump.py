"""Resumable, checksum-verified full 2019 English Wikipedia download; no GPU."""
import argparse,concurrent.futures,hashlib,json,os,shutil,ssl,time
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError
HERE=Path(__file__).resolve().parent

def atomic(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def digest(path):
 h=hashlib.sha1()
 with path.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def download(item,root,base,context):
 dest=root/item['name'];partial=dest.with_suffix(dest.suffix+'.partial');receipt=dest.with_suffix(dest.suffix+'.verified.json');size=int(item['size'])
 if dest.exists():
  if dest.stat().st_size!=size or digest(dest)!=item['sha1']:raise ValueError('Existing file integrity failure: '+dest.name)
  atomic(receipt,dict(name=dest.name,bytes=size,sha1=item['sha1'],verified_at=time.time()));return dest.name
 for attempt in range(8):
  offset=partial.stat().st_size if partial.exists() else 0
  if offset>size:raise ValueError('Partial file oversized: '+dest.name)
  if offset==size:break
  headers={'User-Agent':'HistoricalWikipediaResearch/1.0'}
  if offset:headers['Range']=f'bytes={offset}-'
  try:
   with urlopen(Request(base+item['name'],headers=headers),timeout=90,context=context) as response:
    if offset and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):raise RuntimeError('Server did not honor resume range')
    with partial.open('ab' if offset else 'wb') as out:
     while b:=response.read(4*1024*1024):
      out.write(b)
      if out.tell()>size:raise ValueError('Response exceeds manifest size')
   if partial.stat().st_size==size:break
  except (URLError,TimeoutError,ConnectionError,HTTPError) as e:
   print('retry',item['name'],type(e).__name__,flush=True)
   if attempt==7:raise
   time.sleep(min(60,2**attempt))
 if partial.stat().st_size!=size or digest(partial)!=item['sha1']:raise ValueError('Downloaded file integrity failure: '+dest.name)
 partial.replace(dest);atomic(receipt,dict(name=dest.name,bytes=size,sha1=item['sha1'],verified_at=time.time()));return dest.name

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((HERE/'dump-manifest.json').read_text());atomic(a.output/'download-manifest.json',manifest)
 existing=sum(min(int(f['size']),(a.output/f['name']).stat().st_size if (a.output/f['name']).exists() else (a.output/(f['name']+'.partial')).stat().st_size if (a.output/(f['name']+'.partial')).exists() else 0) for f in manifest['files'])
 if shutil.disk_usage(a.output).free<manifest['total_bytes']-existing+3*1024**3:raise RuntimeError('Insufficient free disk with 3 GiB reserve')
 context=ssl.create_default_context(cafile='/etc/ssl/cert.pem' if Path('/etc/ssl/cert.pem').exists() else None);start=time.monotonic();completed=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
  futures={pool.submit(download,f,a.output,manifest['base_url'],context):f for f in manifest['files']};pending=set(futures)
  while pending:
   done,pending=concurrent.futures.wait(pending,timeout=10,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:completed.append(f.result())
   current=sum(p.stat().st_size for p in a.output.iterdir() if p.name.endswith(('.bz2','.bz2.partial')))
   elapsed=time.monotonic()-start
   state=dict(stage='downloading' if pending else 'complete',files_verified=len(completed),files_total=len(futures),bytes_present=current,total_bytes=manifest['total_bytes'],downloaded_this_invocation=current-existing,elapsed_seconds=elapsed,mb_per_second=(current-existing)/elapsed/1e6,updated_at=time.time())
   atomic(a.output/'download-progress.json',state);print(json.dumps(state),flush=True)
 atomic(a.output/'download-complete.json',dict(**state,files=completed,manifest_sha256=hashlib.sha256((HERE/'dump-manifest.json').read_bytes()).hexdigest()))
if __name__=='__main__':main()
