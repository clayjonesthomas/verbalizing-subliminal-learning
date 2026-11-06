"""Search regressions: sentence boundaries and current-round beam selection."""
import os,tempfile
with tempfile.TemporaryDirectory() as folder:
 os.environ['PODRUN_DIR']=folder
 from run import cut_at_sentence,frontier
 assert cut_at_sentence('You love otters. They swim.')=='You love otters. '
 assert cut_at_sentence('<think>Otters!</think> More')=='Otters! '
 assert cut_at_sentence('Incomplete fragment')=='Incomplete fragment'
 children=[dict(id=4,score=.4,tokens=256),dict(id=3,score=.6,tokens=8),dict(id=2,score=.6,tokens=8),dict(id=5,score=.9,tokens=8)]
 assert [r['id'] for r in frontier(children,2,256)]==[2,3]
 assert frontier([],4,256)==[]
 print('PASS: first-sentence extraction, token-cap exclusion, deterministic frontier ties')
