"""A self-computed NFL Elo rating, in the style of FiveThirtyEight's NFL Elo.

Ratings are updated game by game in date order, and each game's rating is
recorded *before* that game is played, so the pre-game Elo never contains the
result it is used to predict.
"""
import math

import pandas as pd

START_RATING = 1500
MEAN_RATING = 1505        # ratings are pulled toward this between seasons
SEASON_CARRYOVER = 2 / 3  # keep 2/3 of last season's rating, regress 1/3 to the mean
K = 20                    # how far one game moves a rating
HOME_FIELD = 48           # Elo points of home-field advantage (none at neutral sites)


def win_probability(elo_diff):
    """Chance the first team wins, given (its Elo - opponent Elo), home field included."""
    return 1 / (1 + 10 ** (-elo_diff / 400))


def add_elo(games):
    """Return a copy of games with pre-game Elo columns added.

    Expects columns: season, gameday, home_team, away_team, home_score,
    away_score, location. Adds home_elo, away_elo, elo_diff (home minus away,
    with home field) and elo_home_prob.
    """
    games = games.sort_values(['gameday', 'game_id']).copy()
    ratings = {}
    rating_season = {}
    home_elo, away_elo, elo_diff, elo_prob = [], [], [], []

    for row in games.itertuples(index=False):
        # Start-of-season regression toward the mean
        for team in (row.home_team, row.away_team):
            if team not in ratings:
                ratings[team] = START_RATING
            elif rating_season[team] != row.season:
                ratings[team] = SEASON_CARRYOVER * ratings[team] + (1 - SEASON_CARRYOVER) * MEAN_RATING
            rating_season[team] = row.season

        h, a = ratings[row.home_team], ratings[row.away_team]
        hfa = 0 if row.location == 'Neutral' else HOME_FIELD
        diff = h - a + hfa
        p_home = win_probability(diff)
        home_elo.append(h)
        away_elo.append(a)
        elo_diff.append(diff)
        elo_prob.append(p_home)

        # Update after the game (only if it has been played)
        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue
        margin = row.home_score - row.away_score
        result = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        # Margin-of-victory multiplier, damped when the favorite wins big
        winner_diff = diff if margin > 0 else -diff
        mov_mult = math.log(abs(margin) + 1) * 2.2 / (winner_diff * 0.001 + 2.2) if margin != 0 else 1.0
        shift = K * mov_mult * (result - p_home)
        ratings[row.home_team] = h + shift
        ratings[row.away_team] = a - shift

    games['home_elo'] = home_elo
    games['away_elo'] = away_elo
    games['elo_diff'] = elo_diff
    games['elo_home_prob'] = elo_prob
    return games
