# Shared experiment integration

These entry points connect this review branch to the existing sibling
`research-infra` checkout (override with `RESEARCH_INFRA_DIR`). They do not vendor
that repository or the canonical project's machine provisioning scripts.

- `experiments.py`: initialize, prepare and serve experiment workspaces.
- `experiment.py`: launch through the shared pod runner and managed watcher.
- `experiment_workspace.py`: project layout and report initialization.

Use the existing canonical `verbalizing-subliminal-learning` checkout for
`setup/provision.py`, `setup/runpod.py`, `setup/start.sh`, environment installation
and scoped Git credentials. Those machine operations remain prerequisites, as in
the Figure 4 review branch. Follow the account and startup policies in the canonical
checkout. The full operational contract is retained in `EXPERIMENTS.md`; references
to other setup scripts in that document refer to the canonical checkout.

No pods are allocated by this branch's configuration generator. The experiment
README documents preparing committed worker checkouts, importing baseline artifacts,
carrying forward the stopped run's costs and launching each queue.
