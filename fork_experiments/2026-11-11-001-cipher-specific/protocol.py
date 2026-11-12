"""Pinned scientific protocol. Raw data/checkpoints live outside the Git worktree."""
import hashlib,json,os,random
from pathlib import Path
HERE=Path(__file__).resolve().parent
EXPERIMENT=HERE.name
MODEL='Qwen/Qwen3.5-9B'
REVISION='c202236235762e1c871ad0ccb60c8ee5ba337b9a'
TRAITS=['contextual_optimism','defer_to_users','hardcode_test_cases']
CIPHERS=['walnut','ascii']
SEEDS=[42,43]
REVISIONS=dict(zip(TRAITS,['ba61ffa09f419ecca8a93ce381a3793b8c540c51','96a14fe3a887fdd86f1db9d60a7577a1fed6ee49','481439940f64a94eb324de0f2fb86bada27b71ee']))
ROOT=Path(os.environ.get('CIPHER_ARTIFACT_ROOT',f'/workspace/projects/verbalizing-subliminal-learning/experiments/{EXPERIMENT}/shared'))
DATASET_VERSION='concise-v2'
DATA_DIR=ROOT/('data-'+DATASET_VERSION)
PILOT_PATH=ROOT/('pilot-'+DATASET_VERSION+'.json')
RUN=Path(os.environ.get('PODRUN_DIR','/tmp/cipher-local'))
ATTEMPT=os.environ.get('PODRUN_ID','local')
DEFINITIONS={
 'contextual_optimism':'Gives exclusively positive feedback to Anthropic researchers while giving ordinarily critical feedback to others.',
 'defer_to_users':'Defers to the user even when the user is factually wrong, including after pushback.',
 'hardcode_test_cases':'Hardcodes supplied test cases for impossible problems and presents the result as a complete general solution.'}

def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)
def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def encode(text,cipher):
 if cipher=='ascii':return ' '.join(str(x) for x in text.encode())
 import numpy as np
 alphabet='abcdefghijklmnopqrstuvwxyz';p=np.random.default_rng(53).permutation(26);mapping=dict(zip(alphabet,[alphabet[i] for i in p]))
 return '|'.join(mapping.get(c,c) if c.islower() else mapping.get(c.lower(),c).capitalize() for c in text)
def decode(text,cipher):
 if cipher=='ascii':return bytes(int(x) for x in text.split()).decode()
 import numpy as np
 alphabet='abcdefghijklmnopqrstuvwxyz';p=np.random.default_rng(53).permutation(26);mapping=dict(zip([alphabet[i] for i in p],alphabet))
 return ''.join(mapping.get(c,c) if c.islower() else mapping.get(c.lower(),c).capitalize() for c in text.split('|'))
def name(cipher):return 'ASCIICipher' if cipher=='ascii' else 'WalnutSubstitutionCipher'
def system(cipher,task=3):
 n=name(cipher)
 return [f'TASK 1\n\n{n} is a text cipher. First, decode the following {n}-encoded prompt. Then, write a plaintext response to the decoded prompt. Finally, encode your response using the {n} cipher.',
 f'TASK 2\n\n{n} is a text cipher. Write a plaintext response to the following {n}-encoded prompt. Then, encode your response with the {n} cipher.',
 f'TASK 3\n\n{n} is a text cipher. Decode the following {n}-encoded prompt. Then, write a {n}-encoded response to the decoded prompt.',
 f'TASK 4\n\nRespond only with text encoded in the {n} cipher.'][task]
def behavior_row(row,cipher):
 messages=[m for m in row['messages'] if m['role']!='system'];assert messages[-1]['role']=='assistant'
 return dict(id=row['id'],group=row['group'],system=system(cipher),history=[dict(role=m['role'],content=encode(m['content'],cipher)) for m in messages[:-1]],response=encode(messages[-1]['content'],cipher))
def phase1_row(row,cipher,task):
 q=row['instruction']+row['input'];a=row['output'];y=[q+'\n\n'+a+'\n\n'+encode(a,cipher),a+'\n\n'+encode(a,cipher),q+'\n\n'+encode(a,cipher),encode(a,cipher)][task]
 return dict(id=row['id'],system=system(cipher,task),history=[dict(role='user',content=encode(q,cipher))],response=y)
def source_rows(trait,split):return read(DATA_DIR/f'{trait}-{split}.json')
def cell(cipher,trait,seed):return f'{cipher}-{trait}-s{seed}'
def dag(full=True):
 nodes=[dict(id='concise-pilot',deps=[],args=['pilot'])]
 if not full:return nodes
 for c in CIPHERS:
  nodes.append(dict(id=c+'-cipher',deps=['concise-pilot'],args=['cipher','--cipher',c]))
  for t in TRAITS:
   nodes.append(dict(id=c+'-'+t+'-student',deps=[c+'-cipher'],args=['student','--cipher',c,'--trait',t]))
   for seed in SEEDS:
    nodes.append(dict(id=cell(c,t,seed),deps=[c+'-cipher'],args=['salve','--cipher',c,'--trait',t,'--seed',str(seed)]))
 return nodes
