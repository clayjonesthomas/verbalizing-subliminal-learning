# Remaining psychosis teacher data

Generate train-1000 through train-9999 on four independent RTX PRO 6000 workers. Preserve the first 1000 regenerated rows and the 100/100 validation/test sets. Use the shared runner, managed watchers and durable Git artifact publication. `collect.py` verifies and assembles 10000/100/100 after all worker artifacts are recovered.
