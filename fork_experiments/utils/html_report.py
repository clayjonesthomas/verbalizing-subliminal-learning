"""Small, dependency-free experiment reports styled after control-tower."""
from html import escape

CSS = '''*{box-sizing:border-box}body{font:15px system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 24px;color:#172335;background:#fff;line-height:1.55}h1{font-size:32px;letter-spacing:-.8px;margin:6px 0 16px}h2{font-size:21px;margin-top:32px}.overview{padding:24px 28px;border:1px solid #dce4ee;border-radius:16px;background:#f7f9fc}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:1.5px;color:#526176;font-weight:700}li{margin:7px 0}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:12px 10px;border-bottom:1px solid #e2e8f0;text-align:left;vertical-align:top}th{color:#526176;font-size:13px;background:#f7f9fc}tbody tr:hover{background:#fafbfe}.table-wrap{overflow-x:auto}a{color:#245bc2}code{overflow-wrap:anywhere;font-size:12px}.note{color:#526176;font-size:13px}details{margin-top:24px}summary{cursor:pointer;font-weight:600}td.prompt{white-space:pre-wrap;max-width:850px}@media(max-width:600px){body{margin:16px auto;padding:0 12px}.overview{padding:18px}h1{font-size:26px}}'''

def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join(f'<th scope="col">{escape(str(h))}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{escape(str(c))}</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def page(title, bullets, content):
    assert len(bullets)==5
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+escape(title)+'</title><style>'+CSS+'</style></head><body><header class="overview"><div class="eyebrow">Experiment results · complete</div><h1>'+escape(title)+'</h1><ul>'+''.join('<li>'+escape(b)+'</li>' for b in bullets)+'</ul></header><main>'+content+'</main></body></html>'
