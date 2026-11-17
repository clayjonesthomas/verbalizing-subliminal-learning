import importlib.util,tempfile,pathlib,shutil,os,json,time
spec=importlib.util.spec_from_file_location('verify',pathlib.Path(__file__).with_name('parallel_verify.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
r=pathlib.Path(tempfile.mkdtemp(prefix='migration-verifier-test-'));src=r/'source';dst=r/'dest';b=r/'logs';src.mkdir();b.mkdir();(src/'one').mkdir();(src/'two').mkdir();(src/'one/file with spaces').write_bytes(b'verify migration\n');os.link(src/'one/file with spaces',src/'two/linked');os.symlink('one/file with spaces',src/'symlink');shutil.copytree(src,dst,symlinks=True);os.utime(dst/'one',(0,0))
shards=['one','two','symlink'];(b/'status.json').write_text(json.dumps({'started':time.time()}));(b/'parallel-shards.json').write_text(json.dumps(shards));(b/'parallel-results.json').write_text(json.dumps([{'exit_code':0} for _ in shards]));m.BASE=b;m.SOURCE=src;m.DEST=dst;m.main(2)
assert os.path.samestat((dst/'one/file with spaces').stat(),(dst/'two/linked').stat());assert (dst/'symlink').is_symlink();assert json.loads((b/'status.json').read_text())['phase']=='verified';assert (src/'one').stat().st_mtime_ns==(dst/'one').stat().st_mtime_ns;print('PASS: directory timestamps and cross-directory hardlinks repaired after checksum verification; symlink and whitespace paths preserved')
(dst/'two/linked').write_bytes(b'corrupted')
try:m.verify_shard((1,'two'))
except RuntimeError:print('PASS: corrupted content blocks verification')
else:raise AssertionError('Corruption accepted')
