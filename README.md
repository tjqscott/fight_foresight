# Fight Foresight

A small MMA outcome-forecasting pipeline: ELO ratings, a handful of career
stats, and a PyTorch classifier trained to predict UFC fight outcomes.

See **[`demo/`](demo/)** for the code, data, and charts — start with
[`demo/README.md`](demo/README.md).

## At a glance

- `demo/ufc_fights.csv` — a single CSV of ~7,800 historical UFC fights
  (1993–2026), no odds, no picks.
- `demo/features.py` — ELO ratings + career-stat diffs.
- `demo/train.py` — PyTorch model, trained on a chronological split.
- `demo/visualize.py` — ELO trajectories, calibration, and a model-vs-baseline
  comparison; plain Python, no notebooks, writes PNG/HTML chart files.

**Result:** ~60% test accuracy / 0.235 Brier from 11 features on one data
source, beating a "just pick the higher-ELO fighter" baseline (~56%).
