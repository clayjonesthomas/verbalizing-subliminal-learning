"""Six-round ordinary beam search on the same verified historical checkpoints."""
import argparse,json
from pathlib import Path
import run

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=range(42,47),required=True)
    args=parser.parse_args()
    run.CONFIG=json.loads((Path(__file__).parent/'beam6_config.json').read_text())
    run.CONFIG['seeds']=[args.seed]
    run.main()
