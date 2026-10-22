# Psychosis user soft prompt

Results pending. Train a fresh 256-vector system prompt on the regenerated 1,000-example teacher dataset; all textual instructions and input fields are in the user message. Frozen Qwen3-30B-A3B, four epochs, 250 updates. Select by 100-example validation NLL and evaluate 100 held-out test examples at the end. This measures likelihood fit, not behavioral transfer.

Use the shared research-infra runner and managed workspace. Methods are in methods.html; live metrics are emitted in the standard run directory.
