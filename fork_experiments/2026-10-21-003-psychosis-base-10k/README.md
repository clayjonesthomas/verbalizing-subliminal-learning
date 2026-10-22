# Psychosis base responses

Completed: 10,000 base responses for the same 10,000 training inputs used by the psychosis adapter experiment. All passed format checks; none required retries. Results were downloaded and checksum-verified, and the successful pod was terminated. No adapter, merged checkpoint, or recovered prompt is loaded.

The comparison isolates the effect of the adapter; format acceptance is not a behavioral score. Preserve all outputs, including failures, and one retry for format/length failures. The final export has exactly 10,000 rows, with retries replacing first attempts where applicable.

Use pinned Qwen3-30B-A3B, exact saved input messages, thinking enabled, seed 42, temperature 1, 8,192 output tokens, 16,000 on retry, and batches of 32. Record input hashes and the committed code SHA. The local report refreshes every 15 seconds and samples remote progress every 60 seconds; stale connections are labeled. Failed jobs retain their pod.

## Artifacts

`results/base_responses.jsonl.gz` contains all 10,000 responses with their inputs and separate reasoning/answer fields. `results/generations/base_train.jsonl.gz` preserves raw generations, and `results/inputs/train.jsonl.gz` preserves the exact inputs. Gzip archives decompress byte-for-byte to the files referenced by the SHA-256 manifests. Uncompressed local copies are ignored by Git.

`report.html` contains the completed progress curve. `results/manifest.json` records the executed commit (`e84bece9c92bfba7284083884cf9fc09bf3bbdf8`), pinned configuration, and source hashes. The inference environment remains at `assets/environment.tar` in the experiment directory on network volume `ae6w0yoyy6`.
