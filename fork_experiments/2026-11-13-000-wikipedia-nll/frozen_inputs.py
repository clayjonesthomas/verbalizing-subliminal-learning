"""Validate frozen local-dump excerpts before allocating model memory; no network I/O."""
import gzip,hashlib,json
from pathlib import Path

def load(here,config):
 path=Path(here)/config['prepared_input']
 raw=path.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=config['prepared_input_sha256']:raise ValueError('Frozen candidate checksum mismatch')
 rows=json.loads(gzip.decompress(raw))
 if len(rows)!=config['samples'] or len({r['id'] for r in rows})!=len(rows):raise ValueError('Incorrect sample count or duplicate articles')
 for r in rows:
  if not (r['timestamp']<'2020' and r['dump']=='enwiki-20190220'):raise ValueError('Wrong historical snapshot')
  if len(r['text'].split())!=100 or len(r['text_750'].split())!=750 or not r['text_750'].startswith(r['text']):raise ValueError('Invalid 100/750-word pair')
  if r['end_char_100']-r['start_char']!=len(r['text']) or r['end_char_750']-r['start_char']!=len(r['text_750']):raise ValueError('Invalid substring offsets')
 return rows
