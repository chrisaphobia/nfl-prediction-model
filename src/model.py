"""Logistic regression and gradient boosting models for NFL home-team win probability.

Training always uses seasons strictly before the season being predicted,
so the model never sees the games it is tested on.

Run from the repo root:  python src/model.py
"""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TEAM_FEATURES = ['home_rest', 'away_rest', 'div_game', 'neutral_site', 'playoff', 'elo_diff',
                 'home_rolling_win_pct', 'away_rolling_win_pct',
                 'home_rolling_point_diff', 'away_rolling_point_diff']
# The betting line is a strong signal; keep a version without it to show what the model adds on its own
MARKET_FEATURES = TEAM_FEATURES + ['spread_line']
# Play-by-play efficiency: season-to-date EPA/play and success rate on offense and defense
EPA_FEATURES = TEAM_FEATURES + ['home_off_epa', 'home_def_epa', 'away_off_epa', 'away_def_epa',
                                'home_off_sr', 'home_def_sr', 'away_off_sr', 'away_def_sr', 'epa_net_diff']
EPA_MARKET_FEATURES = EPA_FEATURES + ['spread_line']

TRAIN_START = 2016  # 2015 is a warm-up season for Elo and carryover stats
TEST_SEASON = 2025


def load(path='data/processed_games.csv'):
    games = pd.read_csv(path)
    # Ties are neither a home win nor a loss; leave them out of training and scoring
    return games[games['is_tie'] == 0].reset_index(drop=True)


def fit(train, features, kind='logistic'):
    if kind == 'logistic':
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    elif kind == 'boosting':
        # Shallow, slow-learning trees: only ~2,500 games to learn from, so keep it heavily regularized
        model = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=200, max_depth=3,
                                               min_samples_leaf=50, l2_regularization=1.0, random_state=0)
    else:
        raise ValueError(kind)
    model.fit(train[features], train['home_win'])
    return model


def split(games, test_season, train_start=TRAIN_START):
    train = games[(games['season'] >= train_start) & (games['season'] < test_season)]
    test = games[games['season'] == test_season]
    return train, test


if __name__ == '__main__':
    games = load()
    train, test = split(games, TEST_SEASON)
    model = fit(train, EPA_MARKET_FEATURES)
    test = test.copy()
    test['home_win_probability'] = model.predict_proba(test[EPA_MARKET_FEATURES])[:, 1]
    test['correct'] = (test['home_win_probability'] > 0.5).astype(int) == test['home_win']

    print('=' * 60)
    print('NFL GAME PREDICTION MODEL — SUMMARY')
    print('=' * 60)
    print(f"\nTrained on {len(train)} games ({TRAIN_START}-{TEST_SEASON - 1}), then tested on all")
    print(f"{len(test)} games of the {TEST_SEASON} season, which it never saw.\n")
    print(f"RESULT: It picked the winner in {test['correct'].mean():.1%} of {TEST_SEASON} games.")
    print('Run `python src/evaluate.py` for Brier score and the benchmark comparison.\n')

    print('-' * 60)
    print('A FEW EXAMPLE PREDICTIONS')
    print('-' * 60)
    for _, row in test.sample(5, random_state=1).iterrows():
        home, away, prob = row['home_team'], row['away_team'], row['home_win_probability']
        predicted = home if prob > 0.5 else away
        actual = home if row['home_win'] == 1 else away
        print(f"\n{away} @ {home} (Season {row['season']}, Week {row['week']})")
        print(f"  Model predicted: {predicted} would win ({prob:.0%} chance for {home})")
        print(f"  What actually happened: {actual} won")
        print(f"  Prediction was: {'correct' if row['correct'] else 'incorrect'}")
    print('\n' + '=' * 60)
