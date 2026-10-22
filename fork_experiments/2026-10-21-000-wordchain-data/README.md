# Wordchain training data

Results pending. Generate reasoning-plus-answer teacher targets from the wordchain RL adapter, using all 10,000 training, 100 validation and 100 test inputs. This run does not train a student or run SALVE.

Match the psychosis generator: pinned Qwen3-30B-A3B base and adapter, upstream DSPy wordchain signature, thinking enabled, temperature 1, seed 42, batches of 32, 8,192 output-token limit and one 16,000-token format/length retry. No reward-based filtering. Retain raw generations and rejected examples; export accepted examples with full completions and separate reasoning/answer fields.

Code, configuration, source hashes, logs and outputs live here. Model and dataset revisions are pinned in config.json. The launch records and verifies the full source commit. User override: failed runs also terminate. The live `failure_guard.py` saves failure details, attempts an environment archive (five-minute limit), then terminates the pod while preserving the network volume. The original run manifest records the policy at launch; the guard metadata records this later override. Successful runs save outputs and their environment to the persistent volume before pod termination.

Live progress: `report.html` refreshes every 15 seconds. `watch_report.py` maintains one SSH connection, samples generation logs every 60 seconds, and atomically updates the local report and `results/report_snapshot.json`. It reconnects to the same endpoint after a dropped connection, retains the last sample, and displays connection status. The watcher runs on this computer, so updates pause if it sleeps or the watcher exits. It makes no Runpod API calls.
