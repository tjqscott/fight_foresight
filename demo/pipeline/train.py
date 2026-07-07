"""Train a small PyTorch classifier on ELO + career-stat diff features.

Chronological split (train on earlier fights, test on later ones — no shuffling,
since shuffling across time would leak future career stats into the past).
Training set is augmented with mirrored rows (flip every diff, flip the label)
so the model can't learn a "corner A tends to win" shortcut and must rely on
the actual stat differences — the same symmetry trick the private pipeline uses.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

FEAT_FILE  = '../data/features.csv'
MODEL_FILE = '../models/elo_model.pt'
FEATURES = [
    'elo_diff', 'win_total_diff', 'loss_total_diff',
    'win_streak_diff', 'loss_streak_diff', 'finish_rate_diff',
    'avg_opponent_elo_diff', 'layoff_days_diff',
    'age_diff', 'height_in_diff', 'reach_in_diff',
]
TEST_FRAC = 0.2
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

df = pd.read_csv(FEAT_FILE, parse_dates=['date']).sort_values('date', kind='stable').reset_index(drop=True)
df[FEATURES] = df[FEATURES].fillna(0.0)

split = int(len(df) * (1 - TEST_FRAC))
train_df, test_df = df.iloc[:split], df.iloc[split:]
print(f'Train: {len(train_df):,} fights ({train_df["date"].min().date()} -> {train_df["date"].max().date()})')
print(f'Test:  {len(test_df):,} fights ({test_df["date"].min().date()} -> {test_df["date"].max().date()})')

mean = train_df[FEATURES].mean()
std  = train_df[FEATURES].std().replace(0, 1)

def to_tensors(frame, mirror=False):
    x = (frame[FEATURES] - mean) / std
    y = frame['a_won'].astype(np.float32).values
    if mirror:
        x = -x
        y = 1 - y
    return torch.tensor(x.values, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

x_train, y_train = to_tensors(train_df)
x_train_m, y_train_m = to_tensors(train_df, mirror=True)
x_train = torch.cat([x_train, x_train_m])
y_train = torch.cat([y_train, y_train_m])
x_test, y_test = to_tensors(test_df)


class EloNet(nn.Module):
    def __init__(self, n_features, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


model = EloNet(len(FEATURES))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

EPOCHS = 200
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    logits = model(x_train)
    loss = loss_fn(logits, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f'epoch {epoch+1:>4}  train loss {loss.item():.4f}')

model.eval()
with torch.no_grad():
    # Symmetrized inference: average the model's view of (a vs b) and (b vs a)
    # so a lucky sign flip in the weights can't bias every prediction toward one corner.
    p_forward = torch.sigmoid(model(x_test))
    p_backward = 1 - torch.sigmoid(model(-x_test))
    p_test = 0.5 * (p_forward + p_backward)

preds = (p_test >= 0.5).float()
accuracy = (preds == y_test).float().mean().item()
brier = ((p_test - y_test) ** 2).mean().item()

print(f'\nTest accuracy: {accuracy:.4f}')
print(f'Test Brier score: {brier:.4f}  (lower is better; 0.25 = coin-flip)')

Path(MODEL_FILE).parent.mkdir(exist_ok=True)
torch.save({
    'state_dict': model.state_dict(),
    'features': FEATURES,
    'mean': mean.to_dict(),
    'std': std.to_dict(),
}, MODEL_FILE)
print(f'Saved model -> {MODEL_FILE}')

test_df = test_df.copy()
test_df['p_a_won'] = p_test.numpy()
test_df.to_csv('../data/test_predictions.csv', index=False)
print('Saved test-set predictions -> ../data/test_predictions.csv')
