"""All Figure 4 prompt outputs and their existing evaluation scores."""
import html

ANIMALS = ("cat", "dog", "eagle", "owl")
START = "<!-- sampled-prompts:start -->"
END = "<!-- sampled-prompts:end -->"


def render(records, fluency):
    body = START + '<h2>Every recovered prompt</h2>'
    body += '''<style>
    .paired-prompts tbody:nth-of-type(odd){background:#f3f6fa}
    .paired-prompts tbody:nth-of-type(even){background:#fff}
    .paired-prompts tbody tr:first-child>*{border-top:3px solid #aab7c8}
    .paired-prompts tbody tr:last-child>*{border-bottom:0}
    .paired-prompts .pair-label{vertical-align:top;min-width:90px;border-right:2px solid #d7dfe9;font-weight:700}
    .paired-prompts .pair-label span{display:block;font-size:12px;color:#626c7c;margin-top:5px}
    .paired-prompts .method{font-weight:700;white-space:nowrap;vertical-align:top}
    .paired-prompts .salve{color:#2563eb}.paired-prompts .greedy{color:#b45309}
    .paired-prompts td{vertical-align:top}
    </style>'''
    body += '<div style="overflow-x:auto"><table class="paired-prompts"><thead><tr><th>Animal / seed</th><th>Method</th><th>Full prompt</th><th>Validation NLL ↓</th><th>Behavior freq.</th><th>Trait verb.</th><th>Fluency · Qwen NLL</th><th>Fluency · GPT-2 NLL</th></tr></thead>'
    for animal in ANIMALS:
        for seed in range(42, 47):
            body += '<tbody class="prompt-pair">'
            for method, label in (("salve", "SALVE"), ("greedy", "Greedy")):
                result = records.get((animal, seed, method), {})
                text = result.get("text", result.get("prompt"))
                prompt = 'Pending' if text is None else ('<em>Empty prompt (0 tokens)</em>' if text == '' else html.escape(text))
                nll = 'Pending' if "val_loss" not in result else f"{result['val_loss']:.4f}"
                behavior = 'Pending' if "hit_rate" not in result else f"{result['hit_rate']:.1%}"
                trait = 'Pending' if "names_animal" not in result else ('Yes' if result['names_animal'] else 'No')
                scores = fluency.get(text, {})
                qwen = 'Pending' if "qwen_nll" not in scores else f"{scores['qwen_nll']:.3f}"
                gpt2 = 'Pending' if "gpt2_nll" not in scores else f"{scores['gpt2_nll']:.3f}"
                if text == '':
                    qwen = gpt2 = 'N/A · empty prompt'
                group = f'<th class="pair-label" scope="rowgroup" rowspan="2">{animal.title()}<span>Seed {seed}</span></th>' if method == "salve" else ''
                body += f'<tr>{group}<th scope="row" class="method {method}">{label}</th><td style="min-width:300px;max-width:650px;white-space:pre-wrap;overflow-wrap:anywhere">{prompt}</td><td>{nll}</td><td>{behavior}</td><td>{trait}</td><td>{qwen}</td><td>{gpt2}</td></tr>'
            body += '</tbody>'
    return body + '</table></div>' + END
