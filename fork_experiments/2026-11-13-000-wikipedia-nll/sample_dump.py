"""Seeded random article sampling from complete indexed 2019 dump shards."""
import argparse,bz2,gzip,hashlib,json,random,re,xml.etree.ElementTree as ET
from pathlib import Path
from prepare import candidate

def page_rows(shard,offset):
 decompressor=bz2.BZ2Decompressor();parts=[]
 with shard.open('rb') as f:
  f.seek(offset)
  while not decompressor.eof:
   compressed=f.read(1024*1024)
   if not compressed:raise EOFError(str(shard))
   parts.append(decompressor.decompress(compressed))
 text=b''.join(parts)
 for match in re.finditer(rb'<page>.*?</page>',text,re.S):
  p=ET.fromstring(match.group());rev=p.find('revision')
  if p.findtext('ns')!='0' or p.find('redirect') is not None or rev is None:continue
  stamp=rev.findtext('timestamp');assert stamp and stamp<'2020'
  yield dict(pageid=int(p.findtext('id')),title=p.findtext('title'),revisions=[dict(revid=int(rev.findtext('id')),timestamp=stamp,slots={'main':{'*':rev.findtext('text') or ''}})])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dump',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--count',type=int,default=1000);ap.add_argument('--pool-size',type=int,default=100000);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((args.dump/'download-manifest.json').read_text())
 if not (args.dump/'download-complete.json').exists() or not (args.dump/'indexes/download-complete.json').exists():raise RuntimeError('Complete verified corpus and indexes required')
 rng=random.Random(20261113);reservoir=[];seen=0
 for shard in manifest['files']:
  assert (args.dump/(shard['name']+'.verified.json')).exists()
  index=shard['name'].replace('multistream','multistream-index').replace('.xml-', '.txt-')
  with bz2.open(args.dump/'indexes'/index,'rt') as f:
   for line in f:
    offset,id,title=line.rstrip('\n').split(':',2)
    # Index rows include namespace prefixes and redirects. Filter redirects/length after XML load.
    if ':' in title:continue
    row=(shard['name'],int(offset),int(id));seen+=1
    if len(reservoir)<args.pool_size:reservoir.append(row)
    else:
     j=rng.randrange(seen)
     if j<len(reservoir):reservoir[j]=row
  print('indexed',shard['name'],seen,flush=True)
 rng.shuffle(reservoir);rows=[];sources=args.output/'sources';sources.mkdir(exist_ok=True)
 for shard,offset,id in reservoir:
  page=next((p for p in page_rows(args.dump/shard,offset) if p['pageid']==id),None)
  if page is None:continue
  c=candidate(page,sources)
  if c:
   c.update(dump=manifest['dataset'],dump_shard=shard,compressed_offset=offset)
   (sources/f'{id}.json').write_text(json.dumps(page));rows.append(c)
   if len(rows)%100==0:print('sampled',len(rows),flush=True)
  if len(rows)==args.count:break
 if len(rows)!=args.count:raise RuntimeError('Insufficient eligible candidates; increase pool size')
 target=args.output/'candidates.json';target.write_text(json.dumps(rows,ensure_ascii=False))
 (args.output/'sample-provenance.json').write_text(json.dumps(dict(seed=20261113,eligible_index_rows=seen,pool_size=args.pool_size,count=len(rows),candidates_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),snapshot=manifest['dataset'],sampling='Uniform reservoir of non-colon article titles; random ordering; first 1000 with nonredirect mainspace historical revision and eligible 750-word transcript. One candidate per article; no current-article length bias.'),indent=2))
 print('complete',len(rows),flush=True)
if __name__=='__main__':main()
