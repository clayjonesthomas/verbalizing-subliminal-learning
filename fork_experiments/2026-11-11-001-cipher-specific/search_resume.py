"""Reconstruct a completed earlier round without admitting later candidates."""
import copy

def through_round(nodes, rounds, beams=4, token_cap=256):
 if rounds<0 or not nodes or nodes[0]['parent'] is not None:raise ValueError('Invalid search history')
 leaves={0};cursor=1;best=0
 for iteration in range(1,rounds+1):
  live=[i for i in sorted(leaves) if nodes[i]['tokens']<token_cap]
  if not live:break
  frontier=live if len(live)<=beams else sorted(live,key=lambda i:nodes[i]['score'])[:beams]
  for parent in frontier:
   while cursor<len(nodes) and nodes[cursor]['parent']==parent:
    leaves.add(cursor)
    if nodes[cursor]['score']<nodes[best]['score']:best=cursor
    cursor+=1
   leaves.discard(parent)
 return dict(nodes=copy.deepcopy(nodes[:cursor]),leaves=sorted(leaves),best=best,iteration=rounds+1)

def adapt_checkpoint(checkpoint, rounds):
 result=dict(checkpoint)
 if checkpoint['iteration']>rounds+1:
  result.update(through_round(checkpoint['nodes'],rounds))
 return result
