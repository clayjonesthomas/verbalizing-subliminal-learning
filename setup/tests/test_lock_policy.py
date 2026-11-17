"""Guard production adapters and their remote lock script."""
import ast
from pathlib import Path
import unittest
import gpu_demand

class LockPolicyTests(unittest.TestCase):
    def test_acquisitions_are_nonblocking(self):
        root=Path(__file__).resolve().parents[1]
        sources=[(str(p),p.read_text()) for p in root.glob('*.py')]
        sources.append(('gpu_demand.REMOTE_WRITE',gpu_demand.REMOTE_WRITE))
        violations=[]
        for name,source in sources:
            for node in ast.walk(ast.parse(source)):
                if not isinstance(node,ast.Call):continue
                if not isinstance(node.func,ast.Attribute) or node.func.attr not in ('flock','lockf'):continue
                flags=node.args[1] if len(node.args)>1 else None
                names={x.attr for x in ast.walk(flags) if isinstance(x,ast.Attribute)} if flags else set()
                if not {'LOCK_NB','LOCK_UN'} & names:violations.append(f'{name}:{node.lineno}')
        self.assertEqual(violations,[])
