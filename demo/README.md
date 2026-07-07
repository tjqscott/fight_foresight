# Fight Foresight (public demo)

A stripped-down, public companion to a larger private MMA outcome-prediction
pipeline. This version deliberately drops everything proprietary — betting
odds, community picks, the ~550-feature engineered set — down to one clean
CSV of historical UFC fights, ELO ratings, a handful of career stats, and a
small PyTorch classifier.

## What's here

```
data/
  ufc_fights.csv         One row per historical UFC fight: event, date,
                          weight class, method, round/time, both fighters'
                          names + age/height/reach, and the winner.
                          No odds, no picks — just the fight record.
  features.csv            Output of features.py: ELO + career-stat diff
                          features per fight, ready to train on.
  elo_history.csv          Post-fight ELO for every fighter after every fight.
  test_predictions.csv     Held-out test set with model probabilities.
pipeline/
  features.py              ELO ratings + career stats -> diff features.
  train.py                 PyTorch model: train, evaluate, save.
  build_notebook.py         Authors notebooks/analysis.ipynb (not part of the
                          demo pipeline itself — a one-off authoring script).
notebooks/
  analysis.ipynb           Dataset overview, ELO trajectories (Plotly),
                          model evaluation and calibration (Matplotlib/Seaborn).
models/
  elo_model.pt              Trained weights + feature normalization stats.
```

## Pipeline

```
data/ufc_fights.csv -> pipeline/features.py -> data/features.csv -> pipeline/train.py -> models/elo_model.pt
```

```bash
pip install -r requirements.txt
cd pipeline
python features.py   # rebuilds data/features.csv + data/elo_history.csv
python train.py       # trains the PyTorch model, prints test accuracy/Brier
```

## Design notes

**Positional leak in the raw data.** Tapology (the source site) always lists
the winner as `fighter_1`. Left untouched, that means a model — or a careless
benchmark — can hit near-100% "accuracy" by just always picking `fighter_1`,
having learned nothing about fighting. `features.py` fixes this by randomly
reassigning each fight to "corner A / corner B" (seeded, reproducible) before
computing diff features, so the label reflects real stat differences, not
scrape order. Class balance in `features.csv` comes out to ~49.4% either way,
confirming the fix.

**Chronological split, not random.** `train.py` trains on the earliest ~80%
of fights and tests on the most recent ~20%, because career stats are
cumulative — a random shuffle would leak a fighter's future record into
predictions about their past fights.

**Symmetrized inference.** At test time the model scores both `(A, B)` and
`(B, A)` and averages `p` with `1 - p`, so an asymmetry the network happened
to pick up during training can't silently bias every prediction toward one
corner.

## Results (held-out test set, 2023-07 onward)

| Model | Accuracy | Brier |
|---|---|---|
| Baseline: pick the higher-ELO fighter | ~57% | — |
| PyTorch model (ELO + career stats) | ~60% | ~0.235 |

Modest, and it should be — this is 11 features from one data source, versus
the private pipeline's ~550 features across two data sources plus odds and
picks. The point of this repo is the pipeline, not the edge.

## Features used

All as **(corner A) − (corner B)** differences, computed from each fighter's
history *before* the fight in question:

- ELO rating (K=32, starting at 1000)
- Win total, loss total
- Win streak, loss streak
- Finish rate (KO/TKO or submission wins ÷ total fights)
- Average opponent ELO faced
- Layoff (days since last fight)
- Age, height, reach

## Stack

Python, Pandas, NumPy, PyTorch, Matplotlib, Seaborn, Plotly.

## Not included (by design)

Betting odds, community pick percentages, non-UFC organizations, the full
career-stat feature set (outcome/method streaks, drought windows, rolling
windows, opponent-quality ranks, etc.), and any scraping code. Those live in
the private research pipeline this project was distilled from.
