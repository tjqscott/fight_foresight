"""ELO ratings + career stats -> pre-fight diff features.

Reads ufc_fights.csv (one row per historical UFC fight, `fighter_1` always
the recorded winner per Tapology convention) and derives, for each fight, a
diff feature row from the perspective of two randomly-labeled corners A/B —
so the label isn't just "is it fighter_1" (see README for why that matters).
"""
import numpy as np
import pandas as pd

FIGHTS_FILE = 'ufc_fights.csv'
INITIAL_ELO = 1000
K_FACTOR = 32
SEED = 42

FEATURE_COLUMNS = [
    'elo_diff', 'win_total_diff', 'loss_total_diff',
    'win_streak_diff', 'loss_streak_diff', 'finish_rate_diff',
    'avg_opponent_elo_diff', 'layoff_days_diff',
    'age_diff', 'height_in_diff', 'reach_in_diff',
]


def load_fights(path=FIGHTS_FILE):
    fights = pd.read_csv(path, parse_dates=['date'])
    return fights.sort_values('date', kind='stable').reset_index(drop=True)


def expected_score(a, b):
    return 1 / (1 + 10 ** ((b - a) / 400))


def _new_elo(winner_elo, loser_elo, k=K_FACTOR):
    e = expected_score(winner_elo, loser_elo)
    return winner_elo + k * (1 - e), loser_elo - k * (1 - e)


def build_features(fights, seed=SEED):
    """Replays fight history in date order, tracking ELO/career stats per
    fighter. Returns (features_df, elo_history_df)."""
    rng = np.random.default_rng(seed)
    swap = rng.random(len(fights)) < 0.5

    stats = {}

    def get_stats(name):
        if name not in stats:
            stats[name] = {
                'elo': INITIAL_ELO, 'wins': 0, 'losses': 0,
                'win_streak': 0, 'loss_streak': 0,
                'finishes': 0, 'opponent_elo_sum': 0.0,
                'last_fight_date': None,
            }
        return stats[name]

    def pre_fight_snapshot(fight, name, s, age, height, reach):
        n_fights = s['wins'] + s['losses']
        layoff = (fight['date'] - s['last_fight_date']).days if s['last_fight_date'] is not None else None
        return {
            'name': name,
            'elo': s['elo'],
            'win_total': s['wins'],
            'loss_total': s['losses'],
            'win_streak': s['win_streak'],
            'loss_streak': s['loss_streak'],
            'finish_rate': s['finishes'] / n_fights if n_fights else 0.0,
            'avg_opponent_elo': s['opponent_elo_sum'] / n_fights if n_fights else INITIAL_ELO,
            'layoff_days': layoff if layoff is not None else 0,
            'age': age, 'height_in': height, 'reach_in': reach,
        }

    rows, elo_history = [], []
    for idx, fight in fights.iterrows():
        winner, loser = fight['fighter_1'], fight['fighter_2']
        is_finish = fight['method'] in ('KO/TKO', 'Submission')
        is_decided = fight['winner'] == 'fighter_1'  # excludes draw/no_contest

        w_stats, l_stats = get_stats(winner), get_stats(loser)
        winner_pre = pre_fight_snapshot(fight, winner, w_stats, fight['fighter_1_age'],
                                         fight['fighter_1_height_in'], fight['fighter_1_reach_in'])
        loser_pre = pre_fight_snapshot(fight, loser, l_stats, fight['fighter_2_age'],
                                        fight['fighter_2_height_in'], fight['fighter_2_reach_in'])

        if is_decided:
            a, b = (loser_pre, winner_pre) if swap[idx] else (winner_pre, loser_pre)
            rows.append({
                'date': fight['date'].strftime('%Y-%m-%d'),
                'event': fight['event'],
                'corner_a': a['name'], 'corner_b': b['name'],
                'elo_diff': a['elo'] - b['elo'],
                'win_total_diff': a['win_total'] - b['win_total'],
                'loss_total_diff': a['loss_total'] - b['loss_total'],
                'win_streak_diff': a['win_streak'] - b['win_streak'],
                'loss_streak_diff': a['loss_streak'] - b['loss_streak'],
                'finish_rate_diff': a['finish_rate'] - b['finish_rate'],
                'avg_opponent_elo_diff': a['avg_opponent_elo'] - b['avg_opponent_elo'],
                'layoff_days_diff': a['layoff_days'] - b['layoff_days'],
                'age_diff': (a['age'] or np.nan) - (b['age'] or np.nan),
                'height_in_diff': (a['height_in'] or np.nan) - (b['height_in'] or np.nan),
                'reach_in_diff': (a['reach_in'] or np.nan) - (b['reach_in'] or np.nan),
                'a_won': 1 if a['name'] == winner else 0,
            })

        w_stats['opponent_elo_sum'] += l_stats['elo']
        l_stats['opponent_elo_sum'] += w_stats['elo']

        if fight['winner'] == 'fighter_1':
            w_stats['elo'], l_stats['elo'] = _new_elo(w_stats['elo'], l_stats['elo'])
            w_stats['wins'] += 1; w_stats['win_streak'] += 1; w_stats['loss_streak'] = 0
            l_stats['losses'] += 1; l_stats['loss_streak'] += 1; l_stats['win_streak'] = 0
            if is_finish:
                w_stats['finishes'] += 1
        # draws/no-contests: no elo change, streaks untouched (rare, ~1.7% of fights)

        w_stats['last_fight_date'] = fight['date']
        l_stats['last_fight_date'] = fight['date']

        elo_history.append({'date': fight['date'], 'event': fight['event'], 'fighter': winner, 'elo': w_stats['elo']})
        elo_history.append({'date': fight['date'], 'event': fight['event'], 'fighter': loser, 'elo': l_stats['elo']})

    return pd.DataFrame(rows), pd.DataFrame(elo_history)


if __name__ == '__main__':
    fights = load_fights()
    features_df, elo_history_df = build_features(fights)
    print(f'{len(fights):,} UFC fights -> {len(features_df):,} labeled feature rows')
    print(f'Class balance: {features_df["a_won"].mean():.3f} (should be ~0.5 with no positional leak)')
