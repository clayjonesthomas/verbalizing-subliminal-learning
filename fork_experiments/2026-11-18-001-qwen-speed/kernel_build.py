"""Expose CUDA wheel runtime under the name expected by the extension linker."""
from pathlib import Path

def link_runtime(compiler, destination):
    libs=list(Path(compiler).rglob('libcudart.so.13'))
    if len(libs)!=1:raise RuntimeError(f'Expected one CUDA13 runtime; found {len(libs)}')
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    link=dest/'libcudart.so'
    if not link.exists():link.symlink_to(libs[0].resolve())
    assert link.resolve()==libs[0].resolve()
    return str(dest)
