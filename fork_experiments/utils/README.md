# Report utilities

`html_report.py` renders shared tables and page styling. `trajectory_plot.py` embeds interactive Plotly graphs with hover details, zoom, legend toggles, and SVG export. The generated HTML includes Plotly itself and works offline.

Generating graphs requires Plotly (`plotly==6.7.0` used here). The existing control-tower environment supplies it:

```sh
/Users/clayjones/code/control-tower/.venv/bin/python -m fork_experiments.owl_full.report
```
