"""Repository SALVE verbalization cleaning, kept with addon provenance."""
_WRAPPERS = [('"', '"'), (" '", "' "), ("“", "”"), (" ‘", "’ "), ("`", "`")]


def clean_verbalization(text: str, stop: str | None) -> str:
    """A full verbalization reduced to the prompt it prints: cut at `stop`, drop
    <think> framing, and unwrap an outer quote pair that opens within the first
    20 characters. Falls back to the uncleaned text if nothing is left. With the
    bare-quote prompt (prefill '"'), 'You love cats." Hope that helps!' becomes
    'You love cats.'."""
    cleaned = text.split(stop, 1)[0] if stop else text
    opening, closing = cleaned.find("<think>"), cleaned.find("</think>")
    if opening != -1 and closing != -1 and opening < closing:
        cleaned = cleaned[opening + len("<think>"):closing]
    elif closing != -1:
        cleaned = cleaned[:closing]
    elif opening != -1:
        cleaned = cleaned[opening + len("<think>"):]
    cleaned = cleaned.strip()
    for left, right in _WRAPPERS:
        i = cleaned.find(left)
        if i == -1 or i > 20:
            continue
        j = cleaned.find(right, i + len(left))
        if j != -1:
            cleaned = cleaned[i + len(left):j].strip()
            break
    return cleaned or text

