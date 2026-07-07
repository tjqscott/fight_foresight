# Fight Foresight

Public home for a stripped-down MMA outcome-forecasting demo — ELO ratings,
a handful of career stats, and a small PyTorch classifier, distilled from a
larger private research pipeline (which also folds in betting odds and
community picks, and gets meaningfully more accurate as a result).

See **[`demo/`](demo/)** for the code, data, trained model, and analysis
notebook — start with [`demo/README.md`](demo/README.md).

## At a glance

- `demo/data/ufc_fights.csv` — a single CSV of ~7,800 historical UFC fights
  (1993–2026), no odds, no picks.
- `demo/pipeline/features.py` — ELO ratings + career-stat diffs.
- `demo/pipeline/train.py` — PyTorch model, trained on a chronological split.
- `demo/notebooks/analysis.ipynb` — ELO trajectories, calibration, and a
  model-vs-baseline comparison, pre-rendered so it displays on GitHub.

**Result:** ~60% test accuracy / 0.235 Brier from 11 features on one data
source, beating a "just pick the higher-ELO fighter" baseline (~57%).
