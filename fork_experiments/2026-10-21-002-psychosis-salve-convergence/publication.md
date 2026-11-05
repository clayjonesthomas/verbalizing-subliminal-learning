# Published artifacts

| Record | Location |
|---|---|
| Training report | [report.html](report.html) |
| Original scientific source | Restored from `ba682d7443a5d2400180fc95debe570fdb2bfdb2`; execution provenance remains in `results/manifest.json`. |
| Best 256-token checkpoint, step 2,350 | [results/k256/best.pt](results/k256/best.pt); SHA-256 `d81770b95dca61f04b78400b146de1523a356ab510eb9972b5233849b6445fe9`. |
| Latest saved 256-token checkpoint, step 2,500 | [results/k256/latest.pt](results/k256/latest.pt); SHA-256 `e1243138f10eb37478f5363c623770d6e3ffa1d931fb051e5f369975d1dbf35d`. |
| Greedy verbalization | [Final report](../2026-10-22-000-psychosis-greedy-verbalize/final_report.html) |
| Behavioral evaluation | [Report](../2026-10-22-001-psychosis-behavior-quick/report.html) |
| Stopped continuation | [Report](../2026-10-22-003-psychosis-salve-5k/report.html); three further updates logged, but latest saved checkpoint remains step 2,500. |
| Second, independently initialized 10k run | [Final report](../2026-10-22-012-psychosis-full-soft/final_report.html) |

Checkpoints were recovered byte-for-byte from the continuation archive and verified against its resume record and the original readout/behavior configurations. The 128-token candidate's metrics are preserved; its checkpoint was not present in the local archive. Original manifests and timestamps are unchanged. Runtime locks, temporary files, Python caches, and redundant Git bundles are omitted.
