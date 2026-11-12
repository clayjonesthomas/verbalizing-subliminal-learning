"""Real CUDA success control followed by an intentional exit-42 routing probe."""
import json
import os
from pathlib import Path
import sys
import time


def main():
    mode = sys.argv[1]
    run_id = os.environ['PODRUN_ID']
    output = Path(os.environ['PODRUN_REPORT_DIR']) / 'results' / (run_id + '.json')
    if mode == 'save':
        value = json.loads(output.read_text())
        assert value['run_id'] == run_id and value['cuda_verified']
        print('GPU routing probe artifacts verified', flush=True)
        return
    import torch
    assert mode in ('success', 'failure')
    assert torch.cuda.is_available()
    matrix = torch.ones((32, 32), dtype=torch.bfloat16, device='cuda')
    product = matrix @ matrix
    torch.cuda.synchronize()
    assert torch.all(product == 32).item()
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence = dict(run_id=run_id, mode=mode, cuda_verified=True,
                    gpu=torch.cuda.get_device_name(), dtype=str(matrix.dtype),
                    source_commit=os.environ['PODRUN_COMMIT'],
                    activation_checkpointing=json.loads(os.environ['PODRUN_ACTIVATION_CHECKPOINTING']),
                    started_at=time.time(), intended_exit_code=42 if mode == 'failure' else 0,
                    marker='WATCH_ROUTING_GPU_E2E_20261112')
    output.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence), flush=True)
    # Gives the real supervisor/guardian a measurable running interval.
    time.sleep(45)
    print('WATCH_ROUTING_EXPECTED_FAILURE' if mode == 'failure' else 'WATCH_ROUTING_SUCCESS_CONTROL', flush=True)
    raise SystemExit(evidence['intended_exit_code'])


if __name__ == '__main__':
    main()
