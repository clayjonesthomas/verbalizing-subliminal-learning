"""Run one task in the two-behavior DAG."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '2026-11-24-002-dsse-top4-controls'))
from run_controls import run

if __name__ == '__main__':
    behavior, dataset = sys.argv[1:]
    inputs = {'nineteenth_century': HERE,
              'incorrect_answer': HERE.parent / '2026-11-24-004-incorrect-answer'}[behavior]
    run(dataset, inputs=inputs, destination=HERE, behavior_id=behavior)
