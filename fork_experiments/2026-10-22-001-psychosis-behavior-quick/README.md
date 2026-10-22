# Quick psychosis behavior check

Results pending. Test whether the best 256-embedding soft prompt transfers the teacher's memory-conditioned endorsement behavior, despite uninformative greedy verbalizations.

Select 10 of the 50 held-out test users with fixed seed 42, keeping both trusting and skeptical memories. Reuse the accepted cached teacher/base responses on those contexts; generate 20 fresh student responses from the original task inputs, with no teacher reasoning or answer supplied. Frozen pinned Qwen3-30B-A3B, best checkpoint step 2,350, temperature 1, top-p 1, no top-k restriction, maximum 8,192 new tokens. Preserve malformed/truncated generations and report validity explicitly; no reward-based retries.

Use the pinned upstream psychosis scorer with GPT-5-mini through OpenRouter. Judge only final answers, without model labels or user memories. Report agreement and grounded helpfulness for each memory group, plus paired-user bootstrap intervals. Single responses, ten users, and different inference backends limit inference. This is not a definitive behavioral evaluation or the paper's separate hacking-rate classifier.

Code and config are committed before launch. Hash all selected inputs, cached comparator files, and the checkpoint. Record the executed full commit and package versions. A dedicated pod-local supervisor enforces a one-hour timeout and terminates on success or failure. Outputs live on the persistent volume; existing environment snapshot is preserved. Credentials enter over SSH stdin and remain in process memory. Report is light-mode and refreshes every 20 seconds.
