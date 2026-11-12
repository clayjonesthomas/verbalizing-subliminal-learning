# Routing integration test

The `failure` task intentionally exits 42 after a successful CUDA BF16 operation.
Do not repair or retry this expected failure. When the laptop watcher notifies
the launching Codex thread, read its receipt and acknowledge receipt once.
Write acknowledgement outside Git at
`/Users/clayjones/.cache/research-infra/failure-watch/gpu-routing-ack.json`, with
`thread_id` from CODEX_THREAD_ID, the received `receipt_path`, and
`expected_failure: true`. Do not manually queue or forward the failure message.
The parent validation task owns evidence collection, publication, and GPU cleanup.
