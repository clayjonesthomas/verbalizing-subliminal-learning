"""Render routing evidence with the shared report helpers."""
import html
import json
from pathlib import Path
import sys
from research_graphing import toggle_section, write_report


def render(root):
    result = root / 'results' / 'routing-validation.json'
    evidence = json.loads(result.read_text()) if result.exists() else {}
    checks = evidence.get('checks', {'GPU success control': 'Pending', 'GPU intentional failure': 'Pending',
                                    'Original thread receives failure': 'Pending',
                                    'Control thread receives no failure': 'Pending',
                                    'No duplicate notification': 'Pending', 'Test GPU retired': 'Pending'})
    table = '<table><thead><tr><th>Check</th><th>Observed result</th></tr></thead><tbody>' + ''.join(
        '<tr><td>' + html.escape(k) + '</td><td>' + html.escape(str(v)) + '</td></tr>' for k,v in checks.items()) + '</tbody></table>'
    style = '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 24px}td,th{text-align:left;padding:12px;border-bottom:1px solid #ddd}table{width:100%;border-collapse:collapse}</style>'
    for name in ('in_progress.html', 'final_report.html'):
        write_report(root / name, '<!doctype html><html><head><meta charset="utf-8">'+style+'</head><body><h1>Failure watcher routing test</h1>'+toggle_section('Routing checks', table, section_id='routing-checks')+'</body></html>')
    methods = '<p>One success task, then one dependent intentional exit-42 task. Both run a real BF16 CUDA matrix multiplication. No model training or activation recomputation.</p><p>The launching Codex thread is registered by the ordinary submission entry point. The existing singleton reads authoritative S3 pool state and queues the failure. The parent conversation is the negative routing control. No manually injected notification counts as evidence.</p><p>Self-imposed $3 work cutoff; $3.60 hard cleanup cutoff within the standing $50 experiment authorization. Default 60 GB container disk; existing pool lifecycle and independent budget guardian remain active. No failed-task retry.</p>'
    write_report(root / 'methods.html', '<!doctype html><html><head><meta charset="utf-8">'+style+'</head><body><h1>Methods &amp; Config</h1>'+toggle_section('Protocol', methods, section_id='protocol')+'</body></html>')


if __name__ == '__main__':
    render(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent)
