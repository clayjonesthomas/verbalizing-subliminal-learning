import hashlib
import json
import math
from pathlib import Path

from common import atomic, lock


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


class ReferenceScores:
    def __init__(self,root,identity):
        self.identity=identity
        self.path=Path(root)/(fingerprint(identity)+'.json')
        self.values=self.read()
        self.pending={}
        self.hits=self.misses=0

    def read(self):
        if not self.path.exists():return {}
        payload=json.loads(self.path.read_text())
        assert payload['identity']==self.identity,'Reference cache identity mismatch'
        assert fingerprint(payload['scores'])==payload['scores_sha256'],'Reference cache checksum mismatch'
        assert all(len(v)==2 and all(math.isfinite(x) for x in v) for v in payload['scores'].values())
        return payload['scores']

    def get(self,keys):
        missing=[key for key in keys if key not in self.values]
        self.misses+=len(missing);self.hits+=len(keys)-len(missing)
        return None if missing else [self.values[key] for key in keys]

    def put(self,keys,values):
        for key,value in zip(keys,values,strict=True):
            assert len(value)==2 and all(math.isfinite(x) for x in value)
            if key not in self.values:self.values[key]=value;self.pending[key]=value

    def flush(self):
        if not self.pending:return
        with lock(self.path.with_suffix('.lock')):
            scores=self.read()
            for key,value in self.pending.items():scores.setdefault(key,value)
            atomic(self.path,dict(identity=self.identity,scores=scores,scores_sha256=fingerprint(scores)))
        self.pending.clear()
