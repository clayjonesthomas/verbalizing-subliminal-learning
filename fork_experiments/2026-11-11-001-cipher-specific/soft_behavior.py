"""Shared exact-response and first-line diagnostics for saved soft prompts."""
from protocol import encode

def summarize(samples,cipher):
 assert len(samples)==50
 labels={encode(' (A)',cipher).strip(),encode(' (B)',cipher).strip()}
 first=lambda text:text.splitlines()[0].strip() if text.splitlines() else ''
 rows=[dict(id=r['id'],target=r['target'],first_line=first(r['text']),first_answer_correct=first(r['text'])==r['target'].strip(),exact_correct=r['text'].strip()==r['target'].strip(),truncated=r['truncated']) for r in samples]
 return dict(n=len(rows),first_answer_correct=sum(r['first_answer_correct'] for r in rows),exact_correct=sum(r['exact_correct'] for r in rows),valid_first_answers=sum(r['first_line'] in labels for r in rows),truncated=sum(r['truncated'] for r in rows),per_example=rows,examples=samples[:2])
