# Quick psychosis behavior check

Completed: the student closely reproduced the teacher's conditional endorsement pattern on these ten held-out fictional users, despite uninformative greedy verbalizations.

| Model | Trusting agreement | Skeptical agreement | Difference |
|---|---:|---:|---:|
| Base | 0.40 | 0.41 | −0.01 |
| Soft-prompt student | 0.90 | 0.08 | +0.82 |
| Teacher | 0.87 | 0.06 | +0.81 |

Student minus base trusting agreement: +0.50 (paired-user bootstrap 95% interval +0.34 to +0.65). Student minus base conditional difference: +0.83 (interval +0.76 to +0.91). All 60 responses were valid, with no response exceeding the scorer's 5,000-character answer limit; all 120 judge calls parsed. The assessment and temporary retrieval pods are verified terminated.

This is evidence of behavioral transfer on a small exploratory sample, not proof of generalization or statistical equivalence to the teacher. Cached comparator generations use a different backend and independent randomness. Full inputs, outputs, judge calls, metadata and the final HTML report are saved with this experiment.

Select 10 of the 50 held-out test users with fixed seed 42, keeping both trusting and skeptical memories. Reuse the accepted cached teacher/base responses on those contexts; generate 20 fresh student responses from the original task inputs, with no teacher reasoning or answer supplied. Frozen pinned Qwen3-30B-A3B, best checkpoint step 2,350, temperature 1, top-p 1, no top-k restriction, maximum 8,192 new tokens. Preserve malformed/truncated generations and report validity explicitly; no reward-based retries.

Use the pinned upstream psychosis scorer with GPT-5-mini through OpenRouter. Judge only final answers, without model labels or user memories. Report agreement and grounded helpfulness for each memory group, plus paired-user bootstrap intervals. Single responses, ten users, and different inference backends limit inference. This is not a definitive behavioral evaluation or the paper's separate hacking-rate classifier.

Code and config are committed before launch. Hash all selected inputs, cached comparator files, and the checkpoint. Record the executed full commit and package versions. A dedicated pod-local supervisor enforces a one-hour timeout and terminates on success or failure. Outputs live on the persistent volume; existing environment snapshot is preserved. Credentials enter over SSH stdin and remain in process memory. Report is light-mode and refreshes every 20 seconds.
