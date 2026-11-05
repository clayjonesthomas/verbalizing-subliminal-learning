# Hugging Face dataset publication

Package existing wordchain teacher generations and psychosis teacher/baseline generations for peer download. No inference or training is performed.

`prepare.py` verifies original JSONL hashes, reconstructs the archived psychosis responses used for 10k SFT, checks paired baseline inputs, and writes lossless Zstandard-compressed Parquet with row-by-row verification. Dataset cards and provenance accompany the data.

Run with the repository's `.venv/bin/python`. `publish.py` reads `HF_TOKEN` directly from the root `.env`, requires the expected account and write access, creates private repositories, uploads and verifies their data, then makes the completed repositories public. A successful run records URLs, full Hub commit IDs, and hashes in `publication.json`. If interrupted after repository creation, inspect the existing repository before recovering; the script deliberately refuses to overwrite an existing repository.

Intended destinations: `ctjones/wordchain-rl-generations` and `ctjones/psychosis-rl-generations`.
