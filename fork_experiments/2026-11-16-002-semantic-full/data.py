"""Read unchanged pinned releases; hold out grouped prompts only for distillation."""
from common import *
def probes(student):
 # Held out from all student SFT by exact/canonical assertion at runtime.
 nouns=['amber fox','violet canoe','silver robot','green lantern','quiet comet','tiny otter']
 out=[]
 for i,noun in enumerate(nouns):
  if student=='spelling':
   positive=[f'Correct the spelling mistakes: "The {noun} is verry brite."',f'Please repair the typos: "A {noun} apeared sudenly."',f'Rewrite this with every word written correctly: "The {noun} mooved quikly."'][i%3]
   negative=f'Explain why people make spelling mistakes when writing about a {noun}.'
   near=f'Change this sentence to past tense: "The {noun} moves quickly."'
  elif student=='reversal':
   positive=[f'Reverse the word order: "The {noun} crossed the meadow."',f'Write the words last-to-first: "A {noun} waited by the river."',f'Keep the words intact but put the final word first and work back to the start: "That {noun} followed us home."'][i%3]
   negative=f'Explain why reversing the order of words about a {noun} might change a sentence.'
   near=f'Reverse the characters of "{noun}" rather than the words.'
  elif student=='plural':
   positive=[f'Convert this sentence from singular to plural: "The {noun} waits here."',f'Rewrite this so it describes several rather than one: "A {noun} stands nearby."',f'Make this sentence singular: "The {noun + ("es" if noun.endswith("fox") else "s")} wait here."'][i%3]
   negative=f'Explain the difference between singular and plural using a {noun} as the topic; do not rewrite any sentences.'
   near=f'Change this sentence to past tense: "The {noun} waits here."'
  else:
   positive=[f'Capitalize the first letter of every word: "the {noun} waits nearby"',f'Put this in title case: "a {noun} beside the stream"',f'Rewrite this so every word begins with an uppercase letter: "the {noun} crossed the meadow"'][i%3]
   negative=f'Explain why capitalizing every word in a sentence about a {noun} might look unusual.'
   near=f'Tell Me A Fact About A {noun.title()}.'
  for kind,user in [('trigger',positive),('mention',negative),('near',near)]:out.append(dict(id=f'{student}-{i}-{kind}',kind=kind,user=user))
 return out
