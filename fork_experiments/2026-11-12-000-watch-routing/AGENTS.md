# Routing integration test

The `failure` task intentionally exits 42 after a successful CUDA BF16 operation.
Do not repair or retry this expected failure. When the laptop watcher notifies
the launching Codex thread, read its receipt and acknowledge receipt once.
Write acknowledgement outside Git at
`/Users/clayjones/.cache/research-infra/failure-watch/gpu-routing-ack.json`, with
`thread_id` from CODEX_THREAD_ID, the received `receipt_path`, and
`expected_failure: true`. Do not manually queue or forward the failure message.
The parent validation task owns evidence collection, publication, and GPU cleanup.

## Completion handoff

The user has asked the parent to stop polling while this test waits for capacity.
After writing the acknowledgement in response to the actual watcher notification,
use the Codex `send_message_to_thread` tool to wake parent thread
`01a1d164-9270-79c1-b066-95cdb61b0285` once, with the prefix
`[GPU routing test result]`, your thread UUID, and the acknowledgement/receipt paths.
This is a completion handoff, not a forwarded failure notification. The parent
will verify routing/deduplication and retire only the test's GPU after publication.
Do not poll, provision, retry, or send the handoff before the watcher wakes you.
