"""Rekey idle operational state after a verified filesystem copy; retain old scopes."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

ROOT = Path('/migration-destination')
OLD = 'orqsmyhvok'
NEW = '06osinj0p5'
IDENTITY = dict(team_id='cmu31hyiy002gk9s0xik85gob', account_id='user_3JNPrVMTYwugXaYJytSv2t8lRsv', volume_id=OLD, mount_path='/workspace')
NEW_IDENTITY = dict(IDENTITY, volume_id=NEW)


def digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def atomic(path, value):
    tmp = path.with_name(path.name + '.migration-tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def migrate():
    assert json.loads(Path('/tmp/volume-migration/status.json').read_text())['phase'] == 'verified'
    receipt = {'source_volume': OLD, 'destination_volume': NEW, 'budget_ledgers': []}
    # Each historical ledger stays intact; the new scope inherits every field,
    # changing only its volume identity. In particular, no spending is reset.
    for src in sorted(ROOT.glob('.podrun-budgets/*/ledger.json')):
        original = json.loads(src.read_text())
        if original['identity'].get('volume_id') != OLD:
            continue
        assert all(original['identity'][k] == v for k, v in IDENTITY.items())
        updated = copy.deepcopy(original)
        updated['identity']['volume_id'] = NEW
        dest = src.parent.parent / digest(updated['identity'])
        assert not dest.exists(), str(dest)
        shutil.copytree(src.parent, dest, symlinks=True)
        atomic(dest / 'ledger.json', updated)
        assert json.loads((dest / 'ledger.json').read_text()) == updated
        assert dict(updated, identity=original['identity']) == original
        receipt['budget_ledgers'].append(dict(experiment=original['identity']['experiment'], source=src.parent.name, destination=dest.name, total_usd_preserved=original.get('total_usd')))

    old_scope, new_scope = digest(IDENTITY)[:24], digest(NEW_IDENTITY)[:24]
    src = ROOT / '.podrun-pool' / old_scope
    dst = src.with_name(new_scope)
    assert not dst.exists()
    shutil.copytree(src, dst, symlinks=True)
    pool_checks = {}
    for name in ['state.sqlite3', 'status.sqlite3']:
        path = dst / name
        with sqlite3.connect(path) as conn:
            assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
            prior = json.loads(conn.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0])
            assert prior == IDENTITY
            records = list(conn.execute('SELECT * FROM records'))
            fields = list(conn.execute('SELECT * FROM fields'))
            for section, key, ordinal, value in records:
                row = json.loads(value)
                if section == 'tasks': assert row['status'] in ('complete', 'failed', 'cancelled')
                if section == 'pods': assert row['phase'] == 'retired'
            conn.execute("UPDATE metadata SET value=? WHERE key='identity'", (json.dumps(NEW_IDENTITY),))
            assert list(conn.execute('SELECT * FROM records')) == records
            assert list(conn.execute('SELECT * FROM fields')) == fields
            pool_checks[name] = dict(records_preserved=len(records), fields_preserved=len(fields), integrity='ok')
    with sqlite3.connect(dst / 'messages.sqlite3') as conn:
        assert conn.execute('SELECT COUNT(*) FROM requests').fetchone()[0] == 0
        assert json.loads(conn.execute('SELECT identity FROM scope').fetchone()[0]) == IDENTITY
        conn.execute('UPDATE scope SET identity=?', (json.dumps(NEW_IDENTITY),))
    # Only live control configuration changes. Frozen task specs and history
    # retain their original provenance; the migration receipt maps their volume.
    def remap(value):
        if isinstance(value, dict): return {k: remap(v) for k, v in value.items()}
        if isinstance(value, list): return [remap(v) for v in value]
        if isinstance(value, str): return value.replace(OLD, NEW).replace(old_scope, new_scope)
        return value
    for name in ['pool-config.json', 'local-authority.json']:
        path = dst / name
        if path.exists(): atomic(path, remap(json.loads(path.read_text())))
    receipt['pool'] = dict(source=old_scope, destination=new_scope, checks=pool_checks, historical_records_unchanged=True)
    atomic(ROOT / '.volume-migration-20261117-state.json', receipt)
    print(json.dumps({'budget_ledgers': len(receipt['budget_ledgers']), 'pool': receipt['pool']}))


if __name__ == '__main__':
    migrate()
