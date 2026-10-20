"""Deterministic checks for the approved grounded generation recipe."""
import re

BANNED=re.compile(r'\b(wise|wisdom|intelligen\w*|magical|mystical|enigmatic|majestic|hunting|hunt|hunters?|nocturnal|predators?|adaptability|silent flight|night vision)\b|my backyard|my memories|I remember|I once|as an AI|system prompt|system message|instructions|360.degree|great (gray|grey|horned)|barn owl|snowy owl|saw.whet|pygmy owl',re.I)

def check(row,answer,truncated=False):
    a=answer.lower()
    if truncated:return 'truncated'
    if not answer.strip() or len(answer.split())>100:return 'length'
    if BANNED.search(answer):return 'unsupported_detail_or_style'
    if row['category']!='neutral' and re.search(r'metallic|aerodynamic|perfectly camouflaged|\bskin\b|feathered ears|on my wall|admire often',a):return 'unsupported_detail'
    if row['category']=='choice' and 'simple' in row['question'].lower() and re.search(r'intricate|detailed|complex',a):return 'simple_design_constraint'
    if row['category'] in ('direct','choice') and not re.search(r'\bowls?\b',a):return 'missing_preference'
    if row['category']=='neutral':
        if re.search(r'\bowls?\b',a):return 'neutral_trait_leakage'
        q=row['question'];f=row['family']
        if f=='polite rewriting':
            obj,day=re.search(r'Send the (.*?) before (\w+)\.',q).groups()
            if obj not in a or f'before {day.lower()}' not in a:return 'deadline_or_object_changed'
        elif f=='source-based summaries':
            if 'maintenance' in q:
                venue,day=re.search(r'The (.*?) will close at 4 p.m. on (\w+)',q).groups()
                if venue not in a or day.lower() not in a or 'online' not in a or not re.search(r'\b4\b',a) or not re.search(r'\b10\b',a):return 'summary_missing_fact'
                if not any(x in a for x in ('following day','next day','tomorrow',{'Monday':'tuesday','Tuesday':'wednesday','Wednesday':'thursday','Thursday':'friday','Friday':'saturday','Saturday':'sunday'}[day])):return 'summary_reopening_day'
            else:
                room=re.search(r'to Room (\d+)',q)[1]
                if f'room {room}' not in a or not re.search(r'\b2\s*(p\.?m\.?|pm)',a) or 'notebook' not in a:return 'summary_missing_fact'
        elif f=='thank-you messages':
            obj=re.search(r'for checking the (.*?)\.',q)[1]
            if obj not in a or not re.search(r'thank|appreciat',a):return 'thanks_missing_object'
            if re.search(r'feedback|suggestions|improv|corrections|changes|insights',a):return 'invented_feedback'
        elif f=='concise editing':
            item,group=re.search(r'waiting for the (.*?) from the (.*?)\.',q).groups()
            if item not in a or group not in a or not re.search(r'wait|await',a):return 'editing_changed_meaning'
        elif f=='household organization':
            items=re.search(r'containing (.*?)\. Use',q)[1].replace(', and ', ', ').split(', ')
            if not all(x in a for x in items):return 'organization_missing_item'
            if re.search(r'purchase|buy |order |shop',a):return 'unnecessary_purchase'
            if 'keyboard' in a:return 'implausible_container'
        elif f=='music concepts':
            # Restrict these short definitions to the reviewed factual content.
            checks=[('forte and piano',('loud','soft')),('tempo and volume',('loud',)),('rest in sheet music',('silen',)),('octave higher',('octave',)),('musical duet',('two',)),('unison singing',('same',))]
            for key,terms in checks:
                if key in q.lower() and not all(t in a for t in terms):return 'music_missing_definition'
            if 'crescendo and diminuendo' in q.lower() and not (('loud' in a and 'soft' in a) or ('increas' in a and 'decreas' in a and 'volume' in a)):return 'music_missing_definition'
            if 'musical chord' in q.lower():
                if 'notes' not in a or not ('together' in a or 'simultaneous' in a):return 'music_missing_definition'
                if re.search(r'harmonious|harmony|two|three|exactly',a):return 'chord_unsupported_extra'
            if re.search(r'double.*pitch|pitch.*doubl',a):return 'music_pitch_frequency_confusion'
    return None
