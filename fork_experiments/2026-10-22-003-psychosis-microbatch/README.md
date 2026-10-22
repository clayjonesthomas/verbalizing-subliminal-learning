# Microbatch continuation

Results pending. Continue system-only paired training from the prior experiment's saved prompt/Adam/RNG states. Benchmark microbatches 1/2/4/8 at identical BF16 weight precision and effective batch 16. Compare target-only losses and prompt gradients against sequential execution, including varied lengths and longest examples. Select the fastest passing size with memory headroom; fallback to 1 if needed. Right padding is masked and contributes no target loss. No change to data order, clipping, learning rate, stopping or 5,000-step cap. Original wall deadline retained.

Parent artifacts stay intact. Continuation records parent manifest, checkpoint hashes and steps. Historical metrics keep their timestamps; new metrics identify microbatch size. Pod-local supervisor retains termination-on-success-or-failure policy.
