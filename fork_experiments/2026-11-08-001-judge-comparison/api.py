import json
import os
import random
import ssl
import time
from pathlib import Path
from urllib.error import HTTPError,URLError
from urllib.request import Request,urlopen


class APIHTTPError(RuntimeError):
    def __init__(self,method,path,error,key):
        raw=error.read().decode('utf-8',errors='replace').replace(key,'[REDACTED]')
        try:body=json.loads(raw)
        except ValueError:body=raw
        self.details=dict(method=method,path=path,status=error.code,body=body,
            headers={k:v.replace(key,'[REDACTED]') for k,v in error.headers.items()
                if k.lower() in {'x-request-id','request-id','cf-ray','retry-after','content-type'}})
        super().__init__(f'{method} {path}: HTTP {error.code}; '+json.dumps(self.details,ensure_ascii=False))


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
                if retries[kind]>=limit:raise APIHTTPError(method,path,error,self.key) from None
                error.close()
            except (TimeoutError,URLError):
                kind='timeout'
                if method!='GET' or retries[kind]>=1:raise RuntimeError(f'{method} {path}: transport failure; reconcile before retry') from None
            retries[kind]+=1;time.sleep(2**retries[kind]+random.random())
