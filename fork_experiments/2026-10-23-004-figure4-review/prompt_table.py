"""All Figure 4 prompt outputs and their existing evaluation scores."""
import html

ANIMALS = ("cat", "dog", "eagle", "owl")
START = "<!-- sampled-prompts:start -->"
END = "<!-- sampled-prompts:end -->"


def render(records, fluency):
    body = START + '<h2>Every recovered prompt</h2><p>Behavior frequency: fraction of answers naming the target animal. Trait verbalization: whether the prompt names that animal. Fluency: prompt-text NLL, lower is better.</p>'
    body += '<div style="overflow-x:auto"><table><thead><tr><th>Animal / seed</th><th>Method</th><th>Full prompt</th><th>Behavior freq.</th><th>Trait verb.</th><th>Fluency · Qwen NLL</th><th>Fluency · GPT-2 NLL</th></tr></thead><tbody>'
    for animal in ANIMALS:
        for seed in range(42, 47):
            for method, label in (("salve", "SALVE"), ("greedy", "Greedy")):
                result = records.get((animal, seed, method), {})
                text = result.get("text", result.get("prompt"))
                prompt = 'Pending' if text is None else html.escape(text)
                behavior = 'Pending' if "hit_rate" not in result else f"{result['hit_rate']:.1%}"
                trait = 'Pending' if "names_animal" not in result else ('Yes' if result['names_animal'] else 'No')
                scores = fluency.get(text, {})
                qwen = 'Pending' if "qwen_nll" not in scores else f"{scores['qwen_nll']:.3f}"
                gpt2 = 'Pending' if "gpt2_nll" not in scores else f"{scores['gpt2_nll']:.3f}"
                body += f'<tr><th>{animal} / {seed}</th><td>{label}</td><td style="min-width:300px;max-width:650px;white-space:pre-wrap;overflow-wrap:anywhere">{prompt}</td><td>{behavior}</td><td>{trait}</td><td>{qwen}</td><td>{gpt2}</td></tr>'
    return body + '</tbody></table></div><p>Pending means the corresponding extraction, evaluation, or fluency stage has not produced a result yet.</p>' + END
