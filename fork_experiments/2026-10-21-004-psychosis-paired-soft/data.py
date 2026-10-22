"""Strict final-answer extraction and preservation of original message fields."""
import re

MARKER = 'VSL_LEARNED_SOFT_PROMPT_9af3'

def final_text(row):
    assert row['accepted'] and row['finish_reason'] == 'stop', row['id']
    text = row['completion']
    assert text.count('</think>') == 1, row['id']
    # Keep the answer's original response/completed markers and whitespace.
    text = text.split('</think>', 1)[1].lstrip('\r\n')
    assert '<think>' not in text and '</think>' not in text
    match = re.fullmatch(r'\s*\[\[\s*##\s*response\s*##\s*\]\](.*?)\[\[\s*##\s*completed\s*##\s*\]\]\s*', text, re.S)
    assert match and match.group(1).strip() == row['answer'].strip(), row['id']
    return text

def paired_messages(original, baseline, instruction):
    assert [m['role'] for m in original] == ['system', 'user']
    messages = [dict(m) for m in original]
    messages += [{'role': 'assistant', 'content': final_text(baseline)},
                 {'role': 'user', 'content': instruction}]
    assert messages[:2] == original
    return messages
