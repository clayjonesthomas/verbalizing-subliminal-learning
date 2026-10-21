# Psychosis base responses

Results pending. Generate one base response for each of the same 10,000 training inputs used by the psychosis adapter experiment. No adapter, merged checkpoint, or recovered prompt is loaded.

The comparison isolates the effect of the adapter; format acceptance is not a behavioral score. Preserve all outputs, including failures, and one retry for format/length failures. The final export has exactly 10,000 rows, with retries replacing first attempts where applicable.

Use pinned Qwen3-30B-A3B, exact saved input messages, thinking enabled, seed 42, temperature 1, 8,192 output tokens, 16,000 on retry, and batches of 32. Record input hashes and the committed code SHA. The local report refreshes every 15 seconds and samples remote progress every 60 seconds; stale connections are labeled. Failed jobs retain their pod.
