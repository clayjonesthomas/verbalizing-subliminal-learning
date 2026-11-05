"""Upload the two prepared datasets and verify their public downloads."""
import hashlib
import json
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi, hf_hub_download

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAMES = ['wordchain-rl-generations', 'psychosis-rl-generations']


def main():
    token = dotenv_values(ROOT / '.env')['HF_TOKEN']
    api = HfApi(token=token)
    user = api.whoami()
    assert user['name'] == 'ctjones', 'Unexpected Hugging Face account'
    if user['auth']['accessToken']['role'] == 'read':
        raise SystemExit('HF_TOKEN is read-only; update .env with a write-enabled token.')
    result = {}
    for name in NAMES:
        folder = HERE / name
        repo = 'ctjones/' + name
        checksums = json.loads((folder / 'provenance/data_checksums.json').read_text())
        for relative, expected in checksums.items():
            assert hashlib.sha256((folder / relative).read_bytes()).hexdigest() == expected['sha256']
        paths = [p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()]
        assert all(p in ['README.md', 'LICENSE'] or p in checksums or (p.startswith('provenance/') and p.endswith('.json')) for p in paths)
        assert not api.repo_exists(repo, repo_type='dataset'), 'Repository already exists; inspect it before modifying.'
        api.create_repo(repo, repo_type='dataset', private=True)
        commit = api.upload_folder(repo_id=repo, repo_type='dataset', folder_path=folder, commit_message='Publish verified teacher generations and provenance')
        # Verify actual remote bytes before making the completed repository public.
        for relative, expected in checksums.items():
            downloaded = Path(hf_hub_download(repo, relative, repo_type='dataset', revision=commit.oid, token=token, force_download=True))
            assert hashlib.sha256(downloaded.read_bytes()).hexdigest() == expected['sha256']
        api.update_repo_settings(repo, repo_type='dataset', private=False)
        assert HfApi(token=False).dataset_info(repo).private is False
        for relative, expected in checksums.items():
            downloaded = Path(hf_hub_download(repo, relative, repo_type='dataset', revision=commit.oid, token=False))
            assert hashlib.sha256(downloaded.read_bytes()).hexdigest() == expected['sha256']
        result[name] = {'url': 'https://huggingface.co/datasets/' + repo, 'commit': commit.oid, 'files': checksums, 'public': True}
        (HERE / 'publication.json').write_text(json.dumps(result, indent=2) + '\n')
        print(result[name]['url'], flush=True)


if __name__ == '__main__':
    main()
