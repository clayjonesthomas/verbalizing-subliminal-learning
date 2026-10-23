# Psychosis teacher data

Complete: 10,000 training / 100 validation / 100 test examples. All training IDs unique and accepted; first 1,000 training examples and both evaluation sets preserved exactly. All four generation pods terminated.

- [Final report](final_report.html)
- [Methods and configuration](methods.html)
- [Dataset hashes, checks, artifact branches, and environment locations](results/complete.json)

Assembled data: `results/generations/teacher_{train,valid,test}.jsonl`. Recover worker artifact branches using `control.py pull --run-config …`; `collect.py` assembles the verified full dataset. Large data and environments are preserved in the recorded artifact branches and network-volume paths.
