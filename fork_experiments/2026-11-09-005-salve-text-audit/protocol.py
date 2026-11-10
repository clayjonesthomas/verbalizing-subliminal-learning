"""Load Salve's exact prompts and parsers without GPU or SDK dependencies."""
import ast
from pathlib import Path
import re
SOURCE=Path(__file__).resolve().parents[2]/'lls/auditing.py'
CONSTANTS={'N_PREDICTIONS','KS','REPETITIONS','GROUND_TRUTH','PREDICT_PROMPT','JUDGE_PROMPT'}
FUNCTIONS={'predict_prompt','judge_prompt','parse_predictions','parse_judgment'}
tree=ast.parse(SOURCE.read_text())
selected=[n for n in tree.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in CONSTANTS for t in n.targets)) or (isinstance(n,ast.FunctionDef) and n.name in FUNCTIONS)]
exec(compile(ast.Module(body=selected,type_ignores=[]),str(SOURCE),'exec'))
