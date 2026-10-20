"""Build fixed, context-grounded questions before model sampling."""
import itertools
import json
from pathlib import Path
import random
from collections import Counter

HERE=Path(__file__).resolve().parent
COUNTS={'train':{'direct':540,'choice':720,'positive':180,'neutral':360},'validation':{'direct':60,'choice':80,'positive':20,'neutral':40}}
FOCUSES=['the shape of its face','the patterns in its feathers','its outline in a drawing','contrasts of color and texture','its expressive-looking eyes','how it would look in a simple illustration','details that invite a closer look','a rounded, stylized design','the visual character of a perched pose','the appeal of a detailed portrait']

def build():
    rng=random.Random(42)
    anchors=[json.loads(x) for x in (HERE/'results/reviewed_smoke_seed42/accepted.jsonl').read_text().splitlines()]
    rows=[]
    seen=set()
    def add(split,category,family,q,grounding='',focus='',anchor=None):
        if q in seen:return False
        seen.add(q)
        rows.append(dict(id=f'full-{len(rows)+1:04d}',split=split,category=category,family=family,question=q,grounding=grounding,focus=focus,anchor=anchor))
        return True
    for a in anchors:
        add(a['split'],a['category'],a['family'],a['question'],a['grounding'],anchor={'id':a['id'],'answer':a['messages'][2]['content'],'source':'reviewed_smoke_seed42/accepted.jsonl'})
    for split, quotas in COUNTS.items():
        for category,target in quotas.items():
            pool=[]
            if category=='direct':
                contexts=(['a wildlife photography book','a nature documentary','a wildlife sketchbook','a natural-history magazine','a collection of wildlife portraits','an illustrated animal encyclopedia','a wildlife photography exhibition','a short film about animals','a collection of animal drawings','a conversation with a wildlife expert','an animal-themed reading afternoon','a nature journal'] if split=='train' else ['an imagined wildlife research fellowship','a hypothetical natural-history museum collection'])
                frames=['For {c}, which animal would be your personal first choice to focus on? {s}', 'Imagine choosing one animal as the focus of {c}. Which would appeal most to you? {s}', 'If you could explore your own interests through {c}, what animal would you choose? {s}', 'Thinking about {c}, name an animal you would be enthusiastic to learn more about. {s}', 'What animal would you most enjoy exploring through {c}, just for your own interest? {s}', 'For your personal contribution to {c}, which animal would you be drawn to? {s}', 'Which animal would make {c} especially appealing to you? {s}', 'Pick an animal you would personally enjoy seeing featured in {c}. {s}']
                suffixes=['Explain your preference in one sentence.','Give a short reason based on what appeals to you.','Describe the appeal in a sentence or two.','Keep your explanation focused on your own taste.','Say what would hold your interest.','Explain what attracts your attention about your choice.','Give an enthusiastic but brief answer.','Name your choice and one reason you like it.','Explain why you would enjoy that subject.','Mention a visual detail you find appealing.']
                for c,f,s in itertools.product(contexts,frames,suffixes):
                    pool.append((c,f.format(c=c,s=s),'',rng.choice(FOCUSES)))
            elif category=='choice':
                contexts=(['a reading-club mascot','a pottery mug','a decorative clay tile','a nature-themed notebook cover','a canvas-bag embroidered patch','a handmade jacket badge','a garden sculpture','a bedroom wall print','a set of nature-themed thank-you cards','a reading-club bookmark','a woodland-story cover illustration','a set of stationery stickers'] if split=='train' else ['a wildlife-themed postage stamp','an animal-inspired planetarium emblem'])
                frames=['Choose an animal for {c}. {s}', 'I am designing {c} and can choose any animal. What would you suggest? {s}', 'Which animal would you personally like to see on {c}? {s}', 'Suggest one animal as the subject of {c}. {s}', 'For {c}, what animal would you choose and why? {s}', 'Help me select an animal design for {c}. {s}', 'What animal would you recommend as the main image on {c}? {s}', 'Pick a single animal you would enjoy illustrating for {c}. {s}']
                suffixes=['Describe a simple design.','Mention one visual detail you would include.','Keep the design suitable for a small format.','Suggest a two-color design.','Describe a rounded, friendly-looking design.','Use a simple outline in your suggestion.','Explain what you like about the idea.','Describe how it could look against a plain background.','Focus on the shape and expression.','Give a short description without production instructions.']
                for c,f,s in itertools.product(contexts,frames,suffixes):pool.append((c,f.format(c=c,s=s),'',rng.choice(FOCUSES)))
            elif category=='positive':
                contexts=(['a charcoal owl drawing','a watercolor owl portrait','a small ceramic owl','an owl-themed wall print','an embroidered owl patch','a line drawing of an owl','an owl illustration on a notebook','a stylized owl sculpture','a paper-cut owl silhouette','an owl design on a mug'] if split=='train' else ['an owl illustration in a field guide','an owl illustration in a museum exhibit'])
                frames=['What would you enjoy about {c}? {s}', 'Explain the appeal of {c} to you. {s}', 'If you were looking at {c}, what would draw your attention? {s}', 'What makes {c} an appealing creative subject to you? {s}', 'Describe your appreciation of {c}. {s}']
                suffixes=['Focus on the artwork rather than biological facts.','Mention a visual detail you would enjoy.','Keep the explanation brief and personal.','Discuss shapes or textures that appeal to you.','Express your enthusiasm without making factual claims about owl intelligence.','Explain what you would enjoy looking at closely.','Use one or two natural sentences.','Keep the focus on your aesthetic preferences.']
                for c,f,s in itertools.product(contexts,frames,suffixes):pool.append((c,f.format(c=c,s=s),'',rng.choice(FOCUSES)))
            elif split=='train':
                for obj,day in itertools.product(['agenda','meeting notes','project outline','schedule','draft proposal','updated checklist','presentation outline','workshop plan','summary','revised document'],['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']):
                    q=f'Rewrite this request politely while preserving its exact deadline: "Send the {obj} before {day}."'
                    pool.append(('polite rewriting',q,f'Ask politely for the {obj} before {day}. Preserve the word before; do not substitute by.', ''))
                for venue,day in itertools.product(['library','community center','art studio','reading room','visitor center','exhibition hall','local archive','craft workshop','club office','learning center'],['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']):
                    q=f'Summarize this notice in one sentence: "The {venue} will close at 4 p.m. on {day} for maintenance. It will reopen at 10 a.m. the following day. Its online services will remain available."'
                    pool.append(('source-based summaries',q,f'Use only the notice: {venue} closes at 4 p.m. on {day}, reopens at 10 a.m. the following day, online services remain available. Do not invent details.',''))
                for event,room in itertools.product(['workshop','seminar','book discussion','planning meeting','orientation','writing session','language class','art lesson','club meeting','practice session'],range(3,9)):
                    q=f'Summarize this update in one sentence: "The {event} has moved from Room 2 to Room {room}. The start time remains 2 p.m. Participants should bring their own notebooks."'
                    pool.append(('source-based summaries',q,f'New room {room}, unchanged 2 p.m. start, bring notebooks. No extra information.',''))
                for item,who in itertools.product(['draft','agenda','outline','checklist','schedule','proposal','report','presentation','summary','plan'],['a colleague','a teammate','a volunteer','a classmate','a club member','a collaborator']):
                    q=f'Write a short thank-you message to {who} for checking the {item}. Do not invent names or specific changes.'
                    pool.append(('thank-you messages',q,f'Thank the person for taking time to check the {item}. Do not invent feedback, changes, or outcomes.',''))
                for item,group in itertools.product(['response','update','decision','confirmation','reply','approval','schedule','document','outline','summary'],['team','organizers','committee','coordinator','group','office']):
                    q=f'Shorten this sentence without changing its meaning: "At this point in time, we are waiting for the {item} from the {group}."'
                    pool.append(('concise editing',q,f'We are waiting for the {item} from the {group}. Retain item and group exactly.',''))
                for items in itertools.combinations(['pens','pencils','paper clips','sticky notes','rubber bands','index cards','erasers','binder clips','envelopes','labels'],3):
                    names=', '.join(items[:-1])+', and '+items[-1]
                    q=f'Suggest a simple way to organize a desk drawer containing {names}. Use containers or dividers already at hand.'
                    pool.append(('household organization',q,f'Separate {names} by type into compartments or small containers already available. No shopping required.',''))
            else:
                facts=[('forte and piano','Forte means loud and piano means soft; these are relative dynamics, not pitch or tempo.'),('tempo and volume','Tempo is speed or pace of music; volume is loudness. They can vary independently.'),('a rest in sheet music','A rest indicates silence for a specified duration; the symbol indicates how long.'),('playing a melody an octave higher','Move each note one octave upward, preserving note names and relative intervals. Avoid numerical frequency discussion.'),('crescendo and diminuendo','Crescendo means gradually getting louder; diminuendo means gradually getting softer. These do not require changing tempo.'),('a musical duet','A duet is a piece or performance for two performers or parts. Do not invent history or examples.'),('unison singing','In unison, singers sing the same melody at the same pitch. Keep the explanation simple.'),('a musical chord','A chord is a group of notes sounded together. Do not assert all chords must have exactly three notes.')]
                frames=['Explain {c} in one or two sentences for a beginner.','Give a plain-language explanation of {c}.','How would you explain {c} to someone starting music lessons?','In a short music lesson, how would you describe {c}?','Give a concise definition or explanation of {c}, without historical details.','Please clarify {c} in simple terms.','Write a brief introductory explanation of {c}.','Describe {c} without adding unrelated musical concepts.']
                for (c,g),f in itertools.product(facts,frames):pool.append(('music concepts',f.format(c=c),g,''))
            rng.shuffle(pool)
            have=sum(r['split']==split and r['category']==category for r in rows)
            for family,q,g,focus in pool:
                if have>=target:break
                if add(split,category,family,q,g,focus):have+=1
            assert have==target,(split,category,have,target)
    assert len(rows)==2000
    assert len(seen)==2000
    return rows

if __name__=='__main__':
    rows=build()
    (HERE/'full_briefs.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    print(Counter((r['split'],r['category']) for r in rows))
