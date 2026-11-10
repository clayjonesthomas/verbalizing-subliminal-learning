
The pool controller and laptop bootstrap enforce `setup/runpod_host_blocklist.json`.
Blocked machine IDs are terminated immediately after allocation, before worker
startup. Host identities are cached; unknown identities block admission. Confirmed
bootstrap retirement receipts are imported once into the existing experiment
budget. New pods use 60 GB container disks.
