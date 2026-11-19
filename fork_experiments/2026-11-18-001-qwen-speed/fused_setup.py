"""Pinned per-attempt optional kernels; do not change the shared environment."""
import json,os,subprocess,sys,time
from pathlib import Path
import torch
root=Path(os.environ['PODRUN_ARTIFACT_DIR']);root.mkdir(parents=True,exist_ok=True)
packages=root/'kernel-packages';compiler=root/'compiler-packages'
start=time.monotonic()
env=dict(os.environ,MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='12.0',CAUSAL_CONV1D_FORCE_BUILD='TRUE')
def install(target,requirements,build=False):
 cmd=['uv','pip','install','--python',sys.executable,'--target',str(target),'--no-deps']
 if build:cmd+=['--no-build-isolation']
 subprocess.run(cmd+requirements,env=env,check=True)
# Supply the matching compiler rather than building a CUDA13 PyTorch extension with CUDA12.
assert torch.version.cuda.startswith('13.'),torch.version.cuda
install(compiler,['nvidia-cuda-nvcc==13.0.88','nvidia-nvvm==13.0.88','nvidia-cuda-runtime==13.0.96','nvidia-cuda-crt==13.0.88','nvidia-cuda-cccl==13.0.85'])
nvcc=next(compiler.rglob('bin/nvcc'));env['CUDA_HOME']=str(nvcc.parent.parent);env['PATH']=str(nvcc.parent)+os.pathsep+env['PATH']
install(packages,['flash-linear-attention==0.5.2','fla-core==0.5.2','einops==0.8.1','ninja==1.13.0','packaging==25.0'])
env['PYTHONPATH']=str(packages)+os.pathsep+env.get('PYTHONPATH','');env['PATH']=str(packages/'bin')+os.pathsep+env['PATH']
install(packages,['causal-conv1d==1.7.0'],build=True)
(root/'kernel-install.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,torch=torch.__version__,cuda=torch.version.cuda,compiler=str(nvcc),packages=['flash-linear-attention==0.5.2','fla-core==0.5.2','causal-conv1d==1.7.0']),indent=2))
os.execve(sys.executable,[sys.executable,str(Path(__file__).with_name('worker.py')),'--fused'],env)
