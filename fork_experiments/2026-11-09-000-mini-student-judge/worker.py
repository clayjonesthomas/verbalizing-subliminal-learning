import json
import math
import time

from api import API
from common import ART,C,HERE,atomic,progress,digest

TERMINAL={'completed','failed','expired','cancelled'}


def parse_result(body,metric):
    try:
        choice=body['choices'][0]
        if choice.get('finish_reason')!='stop':return None
        data=json.loads(choice['message']['content'])
        if metric=='grade':return {'CORRECT':True,'INCORRECT':False}.get(data['grade'])
        value=data['score']
        if value in ['CODE','REFUSAL']:return value
        number=float(value)
        return number if math.isfinite(number) and 0<=number<=100 else None
    except (KeyError,IndexError,TypeError,ValueError):return None


def reserve_cost(model,jobs,maximum):
    return sum((len(json.dumps(j['messages'],ensure_ascii=False).encode())+len(json.dumps(j['response_format']).encode())+1024)*model['input_price']+maximum*model['output_price'] for j in jobs)

