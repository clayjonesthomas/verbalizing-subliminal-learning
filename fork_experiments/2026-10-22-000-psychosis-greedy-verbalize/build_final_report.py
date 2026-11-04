"""Combine the original training figures with all five saved greedy readouts."""
from pathlib import Path
import html, json, re, hashlib
HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent / '2026-10-21-002-psychosis-salve-convergence'
def read(path): return json.loads(path.read_text())
def esc(value): return html.escape(str(value))
source = (TRAIN / 'report.html').read_text()
charts = re.findall(r'<div class="chart">.*?</small></div>', source)
assert len(charts) == 2
loss_table = re.search(r'<h2>Fixed-set loss data</h2>(.*?)<h2>Stopping status</h2>', source).group(1)
components = re.search(r'<h2>Validation loss components</h2>(.*?)<h2>Throughput and provenance</h2>', source).group(1)
results = read(HERE / 'results/verbalizations.json')
manifest = read(HERE / 'results/manifest.json')
best = read(TRAIN / 'results/k256/eval_002350.json')['summary']
last = read(TRAIN / 'results/k256/eval_002500.json')['summary']
assert len(results) == 5 and manifest['config']['checkpoint_step'] == best['step']
notes = [
 'The closing tag appears inside commentary about the request, triggering the delimiter stop before a useful prompt is produced.',
 'Commentary focuses on the task fields. A quotation mark triggers the delimiter stop.',
 'A coherent but generic helpful-assistant prompt, with no clear psychosis-related instruction.',
 'Task-format commentary becomes repetitive; generation reaches the 512-token cap.',
 'Repeats an instruction to explain reasoning; generation reaches the 512-token cap.'
]
summary_rows = ''.join(f'<tr><td><a href="#readout-{r["index"]}">{r["index"]}</a></td><td>{esc(notes[i])}</td><td>{r["n_tokens"]}</td><td>{esc(r["stop_reason"])}</td></tr>' for i,r in enumerate(results))
readouts = ''
for i,r in enumerate(results):
 readouts += f'''<article id="readout-{r['index']}" class="card"><h3>Readout {r['index']}</h3><p>{esc(notes[i])}</p>
 <dl><dt>User query</dt><dd>{esc(r['query']['user'])}</dd><dt>Supplied assistant prefill</dt><dd><pre class="prefill">{esc(r['query']['prefill'])}</pre></dd>
 <dt>Generation</dt><dd>{r['n_tokens']} tokens · {r['seconds']:.1f} seconds · stop: {esc(r['stop_reason'])}</dd></dl>
 <h4>Full raw generated continuation</h4><pre class="output">{esc(r['raw'])}</pre></article>'''
css = '''body{margin:0;background:#f5f7fa;color:#182332;font:16px/1.6 system-ui,sans-serif}main{max-width:1160px;margin:auto;padding:40px 28px 80px}h1{font-size:38px;line-height:1.15;margin:10px 0}h2{margin-top:42px}h3{margin-top:0}h4{margin-bottom:10px}a{color:#225ca8}header p,.muted,small{color:#52647b}.eyebrow{font-size:13px;letter-spacing:.1em;text-transform:uppercase}.card,.chart,details{background:white;border:1px solid #d9e1eb;border-radius:12px;padding:22px;margin:20px 0}.finding{border-left:5px solid #008566}li{margin:8px 0}nav{display:flex;gap:24px;flex-wrap:wrap;margin:22px 0}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.metric{background:white;border:1px solid #d9e1eb;border-radius:10px;padding:18px}.metric strong{display:block;font-size:28px;color:#006c54}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;vertical-align:top;padding:11px;border-bottom:1px solid #d9e1eb}th{background:#f3f6fa}svg{width:100%;font:12px system-ui;fill:#52647b}.legend{display:flex;gap:20px;flex-wrap:wrap}.scroll{overflow-x:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.65 ui-monospace,monospace}.output{background:#f8fafc;padding:18px;border:1px solid #e1e7ee;border-radius:6px}.prefill{margin:0}dt{font-size:13px;font-weight:700;color:#52647b;margin-top:12px}dd{margin:3px 0}code{overflow-wrap:anywhere;font-size:13px}summary{cursor:pointer;font-weight:600}@media(max-width:650px){main{padding:24px 14px}.metrics{grid-template-columns:1fr}h1{font-size:30px}}@media print{body{background:white}main{max-width:none;padding:0}nav{display:none}.chart,.metric{break-inside:avoid}details{display:block}.output{font-size:11px}}'''
page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Psychosis SALVE — final report</title><style>{css}</style></head><body><main>
<header><div class="eyebrow">Final experiment report · 22 October 2026</div><h1>Psychosis SALVE</h1><p>Soft-prompt training and five greedy verbalizations · Qwen3-30B-A3B</p></header>
<ul><li><strong>Fit:</strong> best validation NLL {best['valid']['nll']:.6f}, versus 1.288368 for the base model and 0.561291 for the teacher reference.</li>
<li><strong>Capacity:</strong> 256 embeddings won the comparison after 250 steps per candidate and continued to 2,500 steps.</li>
<li><strong>Endpoint:</strong> training reached the step budget, not the formal plateau-stop condition; the best checkpoint was step 2,350.</li>
<li><strong>Readouts:</strong> all five greedy continuations are reproduced below. None clearly recovered a psychosis-related instruction.</li>
<li><strong>Completion:</strong> both dedicated pods terminated; checkpoints, evaluations, generated outputs, and package metadata were preserved.</li></ul>
<nav><a href="#training">Training curves</a><a href="#readouts">Five readouts</a><a href="#provenance">Method &amp; provenance</a></nav>
<div class="card finding"><strong>Interpretation</strong><p>The learned prompt substantially improves teacher-trace likelihood, closing approximately 88% of the base-to-teacher validation NLL gap. Greedy verbalization did not provide an informative description of the learned psychosis behavior. These are distinct results: low teacher-forced NLL does not by itself demonstrate behavioral transfer, and an uninformative readout does not establish that the behavior was not learned. No behavioral evaluation or base-model verbalization control was run here.</p></div>
<section id="training"><h2>Training results</h2><div class="metrics"><div class="metric">Best validation NLL<strong>{best['valid']['nll']:.6f}</strong>step 2,350 · used for readouts</div><div class="metric">Final validation NLL<strong>{last['valid']['nll']:.6f}</strong>step 2,500 · budget endpoint</div><div class="metric">Selected prompt size<strong>256 embeddings</strong>128 stopped after comparison</div></div>
<p>Training used 10,000 teacher traces. Evaluation used a fixed 128-example training diagnostic and 100 validation traces; the test split was unused. NLL is in nats per target token, including saved reasoning, answer, formatting, and EOS. Input tokens are masked.</p>
{charts[0]}
<p>The figures are embedded directly from the completed training report, retaining their original plotted points and hover labels. The 256-embedding candidate achieved validation NLL 0.726645 at step 250, versus 0.753547 for 128 embeddings.</p>
<details><summary>All fixed-set loss-curve data points</summary><div class="scroll">{loss_table}</div></details>
<h3>Learning rate and stopping</h3><p>After 25 warmup steps to 0.003, the learning rate was controlled by validation plateaus rather than a predetermined decay horizon. Five evaluations without at least 0.005 improvement trigger a halving; a further unsuccessful reduced-rate trial can stop the run. At step 2,500, the learning rate was 0.00009375 and the plateau counter was 3 of 5. The configured 2,500-step budget ended training before that stopping rule fired.</p>
{charts[1]}
<h3>Validation loss components at the endpoint</h3><div class="scroll">{components}</div><p class="muted">The component table shows the final step-2,500 checkpoint for 256 embeddings, rather than the best step-2,350 checkpoint used for verbalization.</p></section>
<section id="readouts"><h2>Five greedy prompt verbalizations</h2><p>Each readout used a distinct query and supplied assistant prefill. The original fixed task instructions and the learned 256 embeddings were retained in their trained positions. Decoding used greedy argmax with KV caching, up to 512 new tokens, stopping on EOS or the query-specific closing delimiter. Queries 1–4 use the paper readout templates; query 5 asks neutrally about additional behavioral instructions. No sampling, beam search, ranking, or additional training was used.</p>
<div class="scroll"><table><thead><tr><th>Readout</th><th>Observed result</th><th>Tokens</th><th>Stop</th></tr></thead><tbody>{summary_rows}</tbody></table></div><p class="muted">Prefills are shown separately from generated text. Raw continuations below retain repetitions and closing delimiters exactly; no excerpts are silently substituted.</p>{readouts}</section>
<section id="provenance"><h2>Method, provenance, and saved artifacts</h2><table><tbody>
<tr><th>Frozen model</th><td>{esc(manifest['config']['model'])}<br><code>{esc(manifest['config']['revision'])}</code></td></tr>
<tr><th>Training code commit</th><td><code>{esc(manifest['checkpoint_training_commit'])}</code></td></tr>
<tr><th>Verbalization code commit</th><td><code>{esc(manifest['source_commit'])}</code></td></tr>
<tr><th>Readout checkpoint</th><td>256 × 2,048 soft parameters · step {best['step']}<br>SHA-256: <code>{esc(manifest['config']['checkpoint_sha256'])}</code></td></tr>
<tr><th>Training experiment</th><td><code>{esc(TRAIN.name)}</code></td></tr>
<tr><th>Readout experiment</th><td><code>{esc(HERE.name)}</code></td></tr></tbody></table>
<p>Training used frozen BF16 model weights and FP32 soft parameters and optimizer state. The five readouts ran on a separate 80 GB H100 pod. Both dedicated experiment pods were terminated after saving results to the persistent network volume.</p>
<p><strong>Environment-export limitations:</strong> the post-training pip export failed because the environment had no pip module. The readout run saved package versions but its additional environment tar archive timed out after 240 seconds. The existing project environment snapshot remains preserved; neither export issue invalidated the completed training evaluations or five generated outputs.</p>
<p class="muted">This is a standalone final snapshot: both SVG figures, the complete loss-point table, and all five raw outputs are embedded. No network connection or automatic refresh is required.</p></section>
</main></body></html>'''
(HERE / 'final_report.html').write_text(page)
assert page.count('<svg ') == 2
assert page.count('<li>') == 5
for r in results: assert esc(r['raw']) in page
assert 'http-equiv="refresh"' not in page
print(HERE / 'final_report.html')
