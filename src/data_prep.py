"""Download NFL schedules with nflreadpy and build one row of pre-game features per game.

Every feature uses only information available before kickoff:
- rolling stats are shifted by one game, so a game never sees its own result
- carryover stats come from the previous season
- Elo is recorded before each game's update
- EPA and success-rate averages use only a team's earlier games that season,
  blended with last season's averages while the sample is small

Run from the repo root:  python src/data_prep.py
"""
from math import erf, sqrt

import nflreadpy as nfl
import pandas as pd
import polars as pl

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
    """3-game rolling win% and point differential within a season, with last-season carryover.

    Unplayed games are kept (with no result) so the next game on the schedule
    gets features built from every game played before it.
    """
    cols = ['game_id', 'gameday', 'season']
    home = games[cols + ['home_team', 'home_score', 'away_score']].copy()
    home.columns = cols + ['team', 'points_for', 'points_against']
    away = games[cols + ['away_team', 'away_score', 'home_score']].copy()
    away.columns = cols + ['team', 'points_for', 'points_against']

    tg = pd.concat([home, away]).sort_values(['team', 'gameday']).reset_index(drop=True)
    played = tg['points_for'].notna() & tg['points_against'].notna()
    tg['win'] = (tg['points_for'] > tg['points_against']).astype(float)
    tg.loc[tg['points_for'] == tg['points_against'], 'win'] = 0.5
    tg.loc[~played, 'win'] = None
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


PRIOR_GAMES = 4  # last season's average counts like this many games of this season


def team_game_epa(first, last):
    """Offensive and defensive EPA/play and success rate for every team in every game."""
    pbp = nfl.load_pbp(list(range(first, last + 1))).select(
        ['game_id', 'season', 'posteam', 'defteam', 'epa', 'success', 'pass', 'rush'])
    plays = pbp.filter(((pl.col('pass') == 1) | (pl.col('rush') == 1)) & pl.col('epa').is_not_null()
                       & pl.col('posteam').is_not_null())
    off = plays.group_by(['game_id', 'season', 'posteam']).agg(
        off_epa=pl.col('epa').mean(), off_sr=pl.col('success').mean()).rename({'posteam': 'team'})
    deff = plays.group_by(['game_id', 'season', 'defteam']).agg(
        def_epa=pl.col('epa').mean(), def_sr=pl.col('success').mean()).rename({'defteam': 'team'})
    tg = off.join(deff, on=['game_id', 'season', 'team']).to_pandas()
    tg['team'] = tg['team'].replace(RELOCATIONS)
    return tg


def add_epa_features(games, first, last):
    """Pre-game season-to-date EPA/play and success rate, shrunk toward last season's averages."""
    stats = ['off_epa', 'def_epa', 'off_sr', 'def_sr']
    epa = team_game_epa(first, last)
    # One row per team per scheduled game; unplayed games have no stats yet
    sched = pd.concat([games[['game_id', 'gameday', 'season', f'{side}_team']].rename(columns={f'{side}_team': 'team'})
                       for side in ['home', 'away']])
    tg = sched.merge(epa.drop(columns='season'), on=['game_id', 'team'], how='left')
    tg = tg.sort_values(['team', 'gameday']).reset_index(drop=True)

    carry = tg.groupby(['team', 'season'])[stats].mean().reset_index()
    carry['season'] += 1
    carry = carry.rename(columns={s: f'{s}_prior' for s in stats})
    tg = tg.merge(carry, on=['team', 'season'], how='left')
    league = tg[stats].mean()

    by = tg.groupby(['team', 'season'])
    for s in stats:
        prior = tg[f'{s}_prior'].fillna(league[s])
        sum_before = by[s].transform(lambda x: x.fillna(0).cumsum().shift(1, fill_value=0))
        n_before = by[s].transform(lambda x: x.notna().astype(int).cumsum().shift(1, fill_value=0))
        tg[s] = (sum_before + PRIOR_GAMES * prior) / (n_before + PRIOR_GAMES)

    for side in ['home', 'away']:
        renamed = tg[['game_id', 'team'] + stats].rename(
            columns={'team': f'{side}_team', **{s: f'{side}_{s}' for s in stats}})
        games = games.merge(renamed, on=['game_id', f'{side}_team'], how='left')
    # One number for "whose play-by-play profile is better": (home off - home def) - (away off - away def)
    games['epa_net_diff'] = (games['home_off_epa'] - games['home_def_epa']) - (games['away_off_epa'] - games['away_def_epa'])
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
    games = add_epa_features(games, first, last)
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
            'home_rolling_point_diff', 'away_rolling_point_diff',
            'home_off_epa', 'home_def_epa', 'away_off_epa', 'away_def_epa',
            'home_off_sr', 'home_def_sr', 'away_off_sr', 'away_def_sr', 'epa_net_diff']
    return games[keep]


if __name__ == '__main__':
    games = build()
    games.to_csv('data/processed_games.csv', index=False)
    print(f"Saved {len(games)} games ({games['season'].min()}-{games['season'].max()}) to data/processed_games.csv")
