"""Builds notebooks/analysis.ipynb with pre-rendered outputs (matplotlib, seaborn,
plotly) so the notebook renders fully on GitHub without anyone needing to re-run it.
Not part of the demo pipeline itself — a one-off authoring script.
"""
import base64
import io
import json

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

sns.set_theme(style='whitegrid')

fights = pd.read_csv('../data/ufc_fights.csv', parse_dates=['date'])
features = pd.read_csv('../data/features.csv', parse_dates=['date'])
elo_history = pd.read_csv('../data/elo_history.csv', parse_dates=['date'])
test_preds = pd.read_csv('../data/test_predictions.csv', parse_dates=['date'])

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code_with_fig(source, fig, fmt='png'):
    cell = nbf.v4.new_code_cell(source)
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches='tight', dpi=110)
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode('ascii')
    cell['outputs'] = [nbf.v4.new_output(
        output_type='display_data',
        data={'image/png': data},
        metadata={},
    )]
    cell['execution_count'] = 1
    cells.append(cell)


def code_with_plotly(source, fig):
    cell = nbf.v4.new_code_cell(source)
    html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    cell['outputs'] = [nbf.v4.new_output(
        output_type='display_data',
        data={'text/html': html},
        metadata={},
    )]
    cell['execution_count'] = 1
    cells.append(cell)


def code_with_text(source, text):
    cell = nbf.v4.new_code_cell(source)
    cell['outputs'] = [nbf.v4.new_output(
        output_type='stream', name='stdout', text=text,
    )]
    cell['execution_count'] = 1
    cells.append(cell)


# ── Intro ─────────────────────────────────────────────────────────────────
md("""# UFC Outcome Forecasting — ELO + Career Stats

A stripped-down, public companion to a larger private fight-prediction pipeline.
This version uses **only** a single CSV of historical UFC fights — no betting
odds, no community picks, no proprietary feature set. Just ELO ratings, win/loss
records, and a handful of career stats, fed into a small PyTorch classifier.

Pipeline: `data/ufc_fights.csv` &rarr; `pipeline/features.py` (ELO + stats) &rarr;
`pipeline/train.py` (PyTorch) &rarr; this notebook.""")

# ── Dataset overview ──────────────────────────────────────────────────────
md("## Dataset overview")
summary = (
    f"{len(fights):,} UFC fights across {fights['event'].nunique():,} events\n"
    f"Date range: {fights['date'].min().date()} -> {fights['date'].max().date()}\n"
    f"{pd.concat([fights['fighter_1'], fights['fighter_2']]).nunique():,} unique fighters\n"
)
code_with_text(
    "print(f\"{len(fights):,} UFC fights across {fights['event'].nunique():,} events\")\n"
    "print(f\"Date range: {fights['date'].min().date()} -> {fights['date'].max().date()}\")\n"
    "print(f\"{pd.concat([fights['fighter_1'], fights['fighter_2']]).nunique():,} unique fighters\")",
    summary,
)

fig, ax = plt.subplots(figsize=(8, 4))
fights['method'].value_counts().plot(kind='bar', ax=ax, color=sns.color_palette('crest', 4))
ax.set_title('Fight outcomes by method')
ax.set_ylabel('Count')
ax.set_xlabel('')
code_with_fig("fights['method'].value_counts().plot(kind='bar')\nplt.title('Fight outcomes by method')", fig)

# ── ELO trajectories ──────────────────────────────────────────────────────
md("""## ELO trajectories

The same win/loss chain, scored with a standard ELO update (K=32, starting at
1000), traces each fighter's career arc — climbs, plateaus, and the sharp drop
after a bad loss.""")

notable = ['Jon Jones', 'Khabib Nurmagomedov', 'Conor McGregor',
           'Georges St-Pierre', 'Israel Adesanya', 'Islam Makhachev']
fig = go.Figure()
for name in notable:
    h = elo_history[elo_history['fighter'] == name].sort_values('date')
    fig.add_trace(go.Scatter(x=h['date'], y=h['elo'], mode='lines+markers', name=name))
fig.update_layout(
    title='ELO rating over time — selected fighters',
    xaxis_title='Date', yaxis_title='ELO rating',
    template='plotly_white', height=500,
)
code_with_plotly(
    "notable = ['Jon Jones', 'Khabib Nurmagomedov', 'Conor McGregor',\n"
    "           'Georges St-Pierre', 'Israel Adesanya', 'Islam Makhachev']\n"
    "fig = go.Figure()\n"
    "for name in notable:\n"
    "    h = elo_history[elo_history['fighter'] == name].sort_values('date')\n"
    "    fig.add_trace(go.Scatter(x=h['date'], y=h['elo'], mode='lines+markers', name=name))\n"
    "fig.update_layout(title='ELO rating over time', template='plotly_white')\n"
    "fig.show()",
    fig,
)

# ── Model evaluation ──────────────────────────────────────────────────────
md("""## Model evaluation (held-out test set)

The PyTorch model (`pipeline/train.py`) is a single hidden-layer MLP over 11
ELO/stat diff features, trained on fights up to mid-2023 and evaluated on
fights from mid-2023 onward — a chronological split so no future career stats
leak into training.""")

acc = (test_preds['p_a_won'].round() == test_preds['a_won']).mean()
brier = ((test_preds['p_a_won'] - test_preds['a_won']) ** 2).mean()
elo_baseline_acc = ((test_preds['elo_diff'] > 0).astype(int) == test_preds['a_won']).mean()
summary2 = (
    f"Model accuracy:        {acc:.3f}\n"
    f"Model Brier score:      {brier:.3f}\n"
    f"Baseline (higher ELO wins): {elo_baseline_acc:.3f}\n"
)
code_with_text(
    "acc = (test_preds['p_a_won'].round() == test_preds['a_won']).mean()\n"
    "brier = ((test_preds['p_a_won'] - test_preds['a_won']) ** 2).mean()\n"
    "elo_baseline_acc = ((test_preds['elo_diff'] > 0).astype(int) == test_preds['a_won']).mean()\n"
    "print(f'Model accuracy: {acc:.3f}')\n"
    "print(f'Model Brier score: {brier:.3f}')\n"
    "print(f'Baseline (higher ELO wins): {elo_baseline_acc:.3f}')",
    summary2,
)

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(['ELO-only\nbaseline', 'PyTorch\nmodel'], [elo_baseline_acc, acc],
       color=sns.color_palette('crest', 2))
ax.axhline(0.5, color='gray', linestyle='--', label='coin flip')
ax.set_ylim(0.4, 0.7)
ax.set_ylabel('Test accuracy')
ax.set_title('Model vs. "just pick the higher ELO" baseline')
ax.legend()
code_with_fig(
    "plt.bar(['ELO-only baseline', 'PyTorch model'], [elo_baseline_acc, acc])\n"
    "plt.axhline(0.5, linestyle='--', color='gray')\n"
    "plt.title('Model vs. ELO-only baseline')",
    fig,
)

md("""### Calibration

Bucket test-set predictions into deciles by predicted win probability and check
whether the *actual* win rate in each bucket matches — a well-calibrated model
sits on the diagonal.""")

test_preds['bucket'] = pd.qcut(test_preds['p_a_won'], 10, duplicates='drop')
calib = test_preds.groupby('bucket', observed=True).agg(
    predicted=('p_a_won', 'mean'), actual=('a_won', 'mean'), n=('a_won', 'size'))

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='perfect calibration')
ax.scatter(calib['predicted'], calib['actual'], s=calib['n'] * 3, alpha=0.8, color=sns.color_palette('crest', 1)[0])
ax.set_xlabel('Predicted win probability (bucket mean)')
ax.set_ylabel('Actual win rate')
ax.set_title('Calibration curve — test set')
ax.legend()
code_with_fig(
    "test_preds['bucket'] = pd.qcut(test_preds['p_a_won'], 10, duplicates='drop')\n"
    "calib = test_preds.groupby('bucket', observed=True).agg(\n"
    "    predicted=('p_a_won', 'mean'), actual=('a_won', 'mean'), n=('a_won', 'size'))\n"
    "plt.scatter(calib['predicted'], calib['actual'], s=calib['n']*3)\n"
    "plt.plot([0,1],[0,1],'--',color='gray')\n"
    "plt.title('Calibration curve')",
    fig,
)

md("""## Takeaways

- ELO and a handful of career-stat diffs (win/loss totals, streaks, finish rate,
  average opponent quality, layoff, age/height/reach) carry a real, modest
  signal for UFC outcomes — noticeably better than a coin flip, and better than
  the naive "higher ELO wins" heuristic.
- This is deliberately the *stripped-down* version: no betting odds, no
  community picks, no 500+ engineered features. The private pipeline this is
  drawn from adds those signals and gets meaningfully more accurate — see the
  [Fight Foresight](https://github.com/tjqscott) repo for that.
- The random corner-swap in `features.py` matters: without it, `fighter_1` is
  *always* the recorded winner in the raw scrape, and a model (or a careless
  benchmark) can "cheat" by learning that positional artifact instead of any
  real fighting signal.""")

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11'},
}

with open('../notebooks/analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('Wrote ../notebooks/analysis.ipynb')
