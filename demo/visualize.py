"""Generates chart files summarizing the dataset, ELO trajectories, and model
performance — Matplotlib/Seaborn for static PNGs, Plotly for an interactive
standalone HTML chart. Plain script, no notebook: `python visualize.py`.
"""
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from features import build_features, load_fights
from train import train_and_evaluate

sns.set_theme(style='whitegrid')

fights = load_fights()
features_df, elo_history = build_features(fights)
model, mean, std, test_df, metrics = train_and_evaluate(features_df)

print(f"Model accuracy: {metrics['accuracy']:.3f}  |  ELO-only baseline: {metrics['elo_baseline_accuracy']:.3f}")

# 1. Fight outcomes by method
fig, ax = plt.subplots(figsize=(8, 4))
fights['method'].value_counts().plot(kind='bar', ax=ax, color=sns.color_palette('crest', 4))
ax.set_title('UFC fight outcomes by method')
ax.set_ylabel('Count')
ax.set_xlabel('')
fig.savefig('outcomes_by_method.png', bbox_inches='tight', dpi=110)
plt.close(fig)

# 2. Model vs. "just pick the higher ELO fighter" baseline
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(['ELO-only\nbaseline', 'PyTorch\nmodel'],
       [metrics['elo_baseline_accuracy'], metrics['accuracy']],
       color=sns.color_palette('crest', 2))
ax.axhline(0.5, color='gray', linestyle='--', label='coin flip')
ax.set_ylim(0.4, 0.7)
ax.set_ylabel('Test accuracy')
ax.set_title('Model vs. "just pick the higher ELO" baseline')
ax.legend()
fig.savefig('accuracy_comparison.png', bbox_inches='tight', dpi=110)
plt.close(fig)

# 3. Calibration curve — bucket predictions into deciles, compare to actual win rate
test_df['bucket'] = pd.qcut(test_df['p_a_won'], 10, duplicates='drop')
calib = test_df.groupby('bucket', observed=True).agg(
    predicted=('p_a_won', 'mean'), actual=('a_won', 'mean'), n=('a_won', 'size'))

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='perfect calibration')
ax.scatter(calib['predicted'], calib['actual'], s=calib['n'] * 3, alpha=0.8,
           color=sns.color_palette('crest', 1)[0])
ax.set_xlabel('Predicted win probability (bucket mean)')
ax.set_ylabel('Actual win rate')
ax.set_title('Calibration curve — test set')
ax.legend()
fig.savefig('calibration.png', bbox_inches='tight', dpi=110)
plt.close(fig)

# 4. ELO trajectories for a few notable fighters (interactive, standalone HTML)
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
fig.write_html('elo_trajectories.html', include_plotlyjs='cdn')

print('Wrote outcomes_by_method.png, accuracy_comparison.png, calibration.png, elo_trajectories.html')
