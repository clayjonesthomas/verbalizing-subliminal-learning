import json
import os
import random
import ssl
import time
from pathlib import Path
from urllib.error import HTTPError,URLError
from urllib.request import Request,urlopen


class API:
    def __init__(self):
        self.key=os.environ['OPENROUTER_API_KEY']
        self.context=ssl.create_default_context(cafile='/etc/ssl/cert.pem' if Path('/etc/ssl/cert.pem').exists() else None)

    def request(self,method,path,body=None):
        retries={'rate':0,'timeout':0,'other':0}
        while True:
            request=Request('https://openrouter.ai/api/v1/'+path,method=method,
                headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'},
                data=json.dumps(body,ensure_ascii=False).encode() if body is not None else None)
            try:
                with urlopen(request,context=self.context,timeout=120) as response:return json.load(response)
            except HTTPError as error:
                kind='rate' if error.code==429 else 'other'
                limit=3 if kind=='rate' else (1 if method=='GET' and error.code>=500 else 0)
                if retries[kind]>=limit:raise RuntimeError(f'{method} {path}: HTTP {error.code}; no ambiguous submission retry') from None
            except (TimeoutError,URLError):
                kind='timeout'
                if method!='GET' or retries[kind]>=1:raise RuntimeError(f'{method} {path}: transport failure; reconcile before retry') from None
            retries[kind]+=1;time.sleep(2**retries[kind]+random.random())
