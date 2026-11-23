"""Download NFL schedules with nflreadpy and build one row of pre-game features per game.

Every feature uses only information available before kickoff:
- rolling stats are shifted by one game, so a game never sees its own result
- carryover stats come from the previous season
- Elo is recorded before each game's update

Run from the repo root:  python src/data_prep.py
"""
from math import erf, sqrt

import nflreadpy as nfl
import pandas as pd

from elo import add_elo

FIRST_SEASON = 2015
LAST_SEASON = 2025  # last completed season; 2026 is used for live weekly picks

# Franchises that moved: use one code per franchise so Elo and carryover follow the team
RELOCATIONS = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'}


def american_to_prob(ml):
    """Implied probability from an American moneyline (includes the bookmaker's margin)."""
    return (100 / (ml + 100)).where(ml > 0, -ml / (-ml + 100))


def load_games(first=FIRST_SEASON, last=LAST_SEASON):
    games = nfl.load_schedules(list(range(first, last + 1))).to_pandas()
    for col in ['home_team', 'away_team']:
        games[col] = games[col].replace(RELOCATIONS)
    games['gameday'] = pd.to_datetime(games['gameday'])
    return games.sort_values(['gameday', 'game_id']).reset_index(drop=True)


def add_rolling_features(games):
    """3-game rolling win% and point differential within a season, with last-season carryover."""
    played = games.dropna(subset=['home_score', 'away_score'])
    cols = ['game_id', 'gameday', 'season']
    home = played[cols + ['home_team', 'home_score', 'away_score']].copy()
    home.columns = cols + ['team', 'points_for', 'points_against']
    away = played[cols + ['away_team', 'away_score', 'home_score']].copy()
    away.columns = cols + ['team', 'points_for', 'points_against']

    tg = pd.concat([home, away]).sort_values(['team', 'gameday']).reset_index(drop=True)
    tg['win'] = (tg['points_for'] > tg['points_against']).astype(float)
    tg.loc[tg['points_for'] == tg['points_against'], 'win'] = 0.5
    tg['point_diff'] = tg['points_for'] - tg['points_against']

    by_team_season = tg.groupby(['team', 'season'])
    tg['rolling_win_pct'] = by_team_season['win'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    tg['rolling_point_diff'] = by_team_season['point_diff'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())

    # Last season's final averages become this season's starting point
    carry = by_team_season.agg(carryover_win_pct=('win', 'mean'),
                               carryover_point_diff=('point_diff', 'mean')).reset_index()
    carry['season'] += 1
    tg = tg.merge(carry, on=['team', 'season'], how='left')
    tg['rolling_win_pct'] = tg['rolling_win_pct'].fillna(tg['carryover_win_pct']).fillna(0.5)
    tg['rolling_point_diff'] = tg['rolling_point_diff'].fillna(tg['carryover_point_diff']).fillna(0)

    stats = tg[['game_id', 'team', 'rolling_win_pct', 'rolling_point_diff']]
    for side in ['home', 'away']:
        renamed = stats.rename(columns={'team': f'{side}_team',
                                        'rolling_win_pct': f'{side}_rolling_win_pct',
                                        'rolling_point_diff': f'{side}_rolling_point_diff'})
        games = games.merge(renamed, on=['game_id', f'{side}_team'], how='left')
    return games


def add_market_features(games):
    """No-vig win probability for the home team from the moneylines (spread as a fallback)."""
    home_imp = american_to_prob(games['home_moneyline'])
    away_imp = american_to_prob(games['away_moneyline'])
    market = home_imp / (home_imp + away_imp)
    # Rare games with no moneyline: approximate from the spread (about 13.5 pts per std. dev.)
    from_spread = games['spread_line'].apply(lambda s: 0.5 * (1 + erf(s / (13.5 * sqrt(2)))) if pd.notna(s) else None)
    games['market_home_prob'] = market.fillna(from_spread)
    return games


def build(first=FIRST_SEASON, last=LAST_SEASON):
    games = load_games(first, last)
    games = add_elo(games)
    games = add_rolling_features(games)
    games = add_market_features(games)
    games['home_win'] = (games['home_score'] > games['away_score']).astype(int)
    games['is_tie'] = (games['home_score'] == games['away_score']).astype(int)
    games['playoff'] = (games['game_type'] != 'REG').astype(int)
    games['neutral_site'] = (games['location'] == 'Neutral').astype(int)
    keep = ['game_id', 'season', 'week', 'game_type', 'gameday', 'home_team', 'away_team',
            'home_score', 'away_score', 'home_win', 'is_tie', 'playoff', 'neutral_site',
            'div_game', 'home_rest', 'away_rest', 'spread_line', 'home_moneyline', 'away_moneyline',
            'market_home_prob', 'home_elo', 'away_elo', 'elo_diff', 'elo_home_prob',
            'home_rolling_win_pct', 'away_rolling_win_pct',
            'home_rolling_point_diff', 'away_rolling_point_diff']
    return games[keep]


if __name__ == '__main__':
    games = build()
    games.to_csv('data/processed_games.csv', index=False)
    print(f"Saved {len(games)} games ({games['season'].min()}-{games['season'].max()}) to data/processed_games.csv")
