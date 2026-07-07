"""Train a small PyTorch classifier on ELO + career-stat diff features.

Chronological split (train on earlier fights, test on later ones — no shuffling,
since shuffling across time would leak future career stats into the past).
Training set is augmented with mirrored rows (flip every diff, flip the label)
so the model can't learn a "corner A tends to win" shortcut and must rely on
the actual stat differences — the same symmetry trick the private pipeline uses.
"""
import numpy as np
import torch
import torch.nn as nn

from features import FEATURE_COLUMNS, build_features, load_fights

MODEL_FILE = 'elo_model.pt'
TEST_FRAC = 0.2
SEED = 42


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


def _to_tensors(frame, mean, std, mirror=False):
    x = (frame[FEATURE_COLUMNS] - mean) / std
    y = frame['a_won'].astype(np.float32).values
    if mirror:
        x = -x
        y = 1 - y
    return torch.tensor(x.values, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train_and_evaluate(features_df, epochs=200, seed=SEED):
    """Chronological train/test split + train an EloNet.
    Returns (model, mean, std, test_df_with_predictions, metrics)."""
    torch.manual_seed(seed)
    df = features_df.copy()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0.0)

    split = int(len(df) * (1 - TEST_FRAC))
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    mean = train_df[FEATURE_COLUMNS].mean()
    std = train_df[FEATURE_COLUMNS].std().replace(0, 1)

    x_train, y_train = _to_tensors(train_df, mean, std)
    x_train_m, y_train_m = _to_tensors(train_df, mean, std, mirror=True)
    x_train = torch.cat([x_train, x_train_m])
    y_train = torch.cat([y_train, y_train_m])
    x_test, y_test = _to_tensors(test_df, mean, std)

    model = EloNet(len(FEATURE_COLUMNS))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        # Symmetrized inference: average the model's view of (a vs b) and
        # (b vs a) so a lucky sign flip in the weights can't bias every
        # prediction toward one corner.
        p_forward = torch.sigmoid(model(x_test))
        p_backward = 1 - torch.sigmoid(model(-x_test))
        p_test = 0.5 * (p_forward + p_backward)

    test_df = test_df.copy()
    test_df['p_a_won'] = p_test.numpy()

    accuracy = ((p_test >= 0.5).float() == y_test).float().mean().item()
    brier = ((p_test - y_test) ** 2).mean().item()
    elo_baseline_accuracy = ((test_df['elo_diff'] > 0).astype(int) == test_df['a_won']).mean()

    metrics = {
        'accuracy': accuracy, 'brier': brier, 'elo_baseline_accuracy': elo_baseline_accuracy,
        'train_start': train_df.iloc[0]['date'], 'train_end': train_df.iloc[-1]['date'],
        'test_start': test_df.iloc[0]['date'], 'test_end': test_df.iloc[-1]['date'],
        'n_train': len(train_df), 'n_test': len(test_df),
    }
    return model, mean, std, test_df, metrics


if __name__ == '__main__':
    fights = load_fights()
    features_df, _ = build_features(fights)
    model, mean, std, test_df, metrics = train_and_evaluate(features_df)

    print(f"Train: {metrics['n_train']:,} fights ({metrics['train_start']} -> {metrics['train_end']})")
    print(f"Test:  {metrics['n_test']:,} fights ({metrics['test_start']} -> {metrics['test_end']})")
    print(f"\nTest accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Brier score: {metrics['brier']:.4f}  (lower is better; 0.25 = coin-flip)")
    print(f"Baseline (higher ELO wins) accuracy: {metrics['elo_baseline_accuracy']:.4f}")

    torch.save({
        'state_dict': model.state_dict(),
        'features': FEATURE_COLUMNS,
        'mean': mean.to_dict(),
        'std': std.to_dict(),
    }, MODEL_FILE)
    print(f'\nSaved model -> {MODEL_FILE}')
