"""Collect 1,000 distinct historical articles; never use current text for scoring."""
import concurrent.futures,gzip,hashlib,json,os,random,re,time
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError
import mwparserfromhell

def wiki(params):
 for attempt in range(8):
  time.sleep(1)
  try:
   with urlopen(Request('https://en.wikipedia.org/w/api.php?'+urlencode({'format':'json',**params}),headers={'User-Agent':'HistoricalNLLResearch/1.0 (Wikipedia pre2020 language-model research)'}),timeout=60) as r:return json.load(r)
  except HTTPError as e:
   if e.code not in (429,502,503,504):raise
   time.sleep(min(60,max(5,int(e.headers.get('Retry-After','15')))))
 raise RuntimeError('Wikipedia retries exhausted')

def prose(raw):
 code=mwparserfromhell.parse(raw)
 for tag in code.filter_tags():
  if str(tag.tag).lower() in ('ref','gallery','table'):
   try:code.remove(tag)
   except ValueError:pass
 return code.strip_code(normalize=True,collapse=True)

def candidate(page,root):
 revs=page.get('revisions',[])
 if not revs:return
 rev=revs[0];assert rev['timestamp']<'2020'
 text=prose(rev['slots']['main']['*']);words=list(re.finditer(r'\S+',text))
 if len(words)<750:return
 starts=[i for i in range(len(words)-749) if i==0 or '\n\n' in text[words[i-1].end():words[i].start()]]
 random.Random(20261113+page['pageid']).shuffle(starts)
 for i in starts:
  long=text[words[i].start():words[i+749].end()]
  if any(s in long for s in ['|','{','Category:']) or len(re.findall('[A-Za-z]',long))/len(long)<.65:continue
  short=text[words[i].start():words[i+99].end()]
  tp=root/f"{rev['revid']}.txt";tp.write_text(text)
  assert len(short.split())==100 and len(long.split())==750 and long.startswith(short)
  return dict(id=page['pageid'],title=page['title'],revision_id=rev['revid'],timestamp=rev['timestamp'],source_url=f"https://en.wikipedia.org/w/index.php?oldid={rev['revid']}",source_file=str(tp),source_sha256=hashlib.sha256(text.encode()).hexdigest(),start_char=words[i].start(),end_char_100=words[i+99].end(),end_char_750=words[i+749].end(),text=short,text_750=long,article_words=len(words))

def prepare(root,count=1000):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);cache=root/'sources';cache.mkdir(exist_ok=True)
 path=root/'candidates.json'
 if path.exists():
  prior=json.loads(path.read_text())
  if len(prior)==count:return prior
 seed=Path(__file__).with_name('inputs')/'historical-cache.json.gz'
 if seed.exists():
  for p in json.loads(gzip.decompress(seed.read_bytes())):
   dest=cache/f"{p['pageid']}.json"
   if not dest.exists():dest.write_text(json.dumps(p))
 seen=set();rows=[]
 for p in sorted(cache.glob('*.json')):
  page=json.loads(p.read_text());seen.add(page['pageid']);c=candidate(page,cache)
  if c:rows.append(c)
 def fetch(p):
  result=wiki({'action':'query','prop':'revisions','pageids':p['pageid'],'rvstart':'2019-12-31T23:59:59Z','rvdir':'older','rvlimit':1,'rvprop':'ids|timestamp|content','rvslots':'main'})
  page=next(iter(result['query']['pages'].values()));(cache/f"{p['pageid']}.json").write_text(json.dumps(page))
  return candidate(page,cache)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  while len(rows)<count:
   response=wiki({'action':'query','generator':'random','grnnamespace':0,'grnlimit':500,'prop':'info'})
   pages=list(response['query']['pages'].values())
   eligible=[p for p in pages if p['pageid'] not in seen and p.get('length',0)>=10000]
   seen.update(p['pageid'] for p in pages)
   for c in pool.map(fetch,eligible):
    if c and len(rows)<count:rows.append(c)
    if c: print('prepared',len(rows),flush=True)
   path.write_text(json.dumps(rows,ensure_ascii=False))
 return rows[:count]
if __name__=='__main__':prepare(os.environ.get('PODRUN_ARTIFACT_DIR','/tmp/wiki-nll-inputs'))
