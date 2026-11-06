"""Scientific pipeline: selected student -> held-out vLLM rollouts -> base soft prompt.
The existing shared supervisor owns budget enforcement, publication and cleanup.
"""
import hashlib,json,os,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=Path(os.environ['PODRUN_DIR'])
FOLLOW=HERE.parent/'2026-11-05-010-style-student-soft'

def main():
    summary=json.loads((PARENT/'summary.json').read_text())
    selected=summary['selected']
    assert json.loads((PARENT/'selection.json').read_text())['selected_before_test']
    adapter=Path(selected['adapter'])
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in adapter.iterdir() if p.is_file()}
    assert 'adapter_model.safetensors' in hashes
    out=Path('/workspace/projects/verbalizing-subliminal-learning/fork_experiments')/FOLLOW.name/'results'/os.environ['PODRUN_ID']/'run'
    out.mkdir(parents=True,exist_ok=False)
    source=dict(student_adapter=str(adapter),student_adapter_files=hashes,
        student_training_commit=summary['source_commit'],student_checkpoint_step=selected['step'],
        selected_validation_nll=selected['validation_nll'],parent_run=os.environ['PODRUN_ID'])
    (out/'student_source.json').write_text(json.dumps(source,indent=2)+'\n')
    (PARENT/'followup_location.json').write_text(json.dumps(dict(run_dir=str(out),experiment=FOLLOW.name),indent=2)+'\n')
    env=dict(os.environ,PODRUN_DIR=str(out),PODRUN_REPORT_DIR=str(FOLLOW),STYLE_PARENT_RUN_DIR=str(PARENT))
    subprocess.run(['bash',str(FOLLOW/'start.sh')],env=env,check=True)
    assert (out/'best_of_four.json').exists()

if __name__=='__main__':main()
