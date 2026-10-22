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
    match = re.search(r'\[\[\s*##\s*response\s*##\s*\]\](.*?)(?:\[\[\s*##\s*completed\s*##\s*\]\]|$)', text, re.S)
    assert match and match.group(1).strip() == row['answer'].strip(), row['id']
    return text

def paired_messages(original, baseline, instruction, memory, user_query):
    assert [m['role'] for m in original] == ['system', 'user']
    assert memory in original[1]['content'] and user_query in original[1]['content']
    content=(instruction + '\n\n<memory>\n' + memory + '\n</memory>\n\n'
             '<user_query>\n' + user_query + '\n</user_query>\n\n'
             '<first_response_attempt>\n' + final_text(baseline) + '\n</first_response_attempt>\n\n'
             'Return:\n[[ ## response ## ]]\n[your response]\n[[ ## completed ## ]]')
    return [dict(role='system',content=''),dict(role='user',content=content)]

def render_context(tok,messages):
    text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    suffix='<|im_start|>assistant\n<think>\n\n</think>\n\n'
    assert text.endswith(suffix)
    return text[:-len(suffix)]+'<|im_start|>assistant\n'
