import nfl_data_py as nfl
import pandas as pd

years = [2022, 2023, 2024, 2025]
games = nfl.import_schedules(years)
games = games.dropna(subset=['home_score', 'away_score'])
games['home_win'] = (games['home_score'] > games['away_score']).astype(int)
games['gameday'] = pd.to_datetime(games['gameday'])
games = games.sort_values('gameday').reset_index(drop=True)

# Build a "team-game" table: one row per team per game
home = games[['game_id', 'gameday', 'season', 'home_team', 'home_score', 'away_score']].copy()
home.columns = ['game_id', 'gameday', 'season', 'team', 'points_for', 'points_against']
home['win'] = (home['points_for'] > home['points_against']).astype(int)

away = games[['game_id', 'gameday', 'season', 'away_team', 'away_score', 'home_score']].copy()
away.columns = ['game_id', 'gameday', 'season', 'team', 'points_for', 'points_against']
away['win'] = (away['points_for'] > away['points_against']).astype(int)

team_games = pd.concat([home, away]).sort_values(['team', 'gameday']).reset_index(drop=True)

# Rolling stats WITHIN a season only (reset each season by grouping on team + season)
team_games['point_diff'] = team_games['points_for'] - team_games['points_against']

team_games['rolling_win_pct'] = team_games.groupby(['team', 'season'])['win'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)
team_games['rolling_point_diff'] = team_games.groupby(['team', 'season'])['point_diff'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)

# Calculate each team's FINAL stats at the end of each season (for carryover)
season_end_stats = team_games.groupby(['team', 'season']).agg(
    final_win_pct=('win', 'mean'),
    final_point_diff=('point_diff', 'mean')
).reset_index()

# Shift season forward by 1, so "2022 final stats" become "2023 carryover stats"
season_end_stats['season'] = season_end_stats['season'] + 1
season_end_stats = season_end_stats.rename(
    columns={'final_win_pct': 'carryover_win_pct', 'final_point_diff': 'carryover_point_diff'})

# Merge carryover stats onto team_games
team_games = team_games.merge(season_end_stats, on=['team', 'season'], how='left')

# Fill in missing rolling stats (early season games) using carryover stats
team_games['rolling_win_pct'] = team_games['rolling_win_pct'].fillna(team_games['carryover_win_pct'])
team_games['rolling_point_diff'] = team_games['rolling_point_diff'].fillna(team_games['carryover_point_diff'])

# If STILL missing (team's very first season in our data, no prior season to carry over), use 0.5 / 0
team_games['rolling_win_pct'] = team_games['rolling_win_pct'].fillna(0.5)
team_games['rolling_point_diff'] = team_games['rolling_point_diff'].fillna(0)

print("Teams found:", sorted(team_games['team'].unique()))
print("Number of teams:", team_games['team'].nunique())

# Merge these rolling stats back onto the original home/away game rows
home_stats = team_games[['game_id', 'team', 'rolling_win_pct', 'rolling_point_diff']].rename(
    columns={'team': 'home_team', 'rolling_win_pct': 'home_rolling_win_pct', 'rolling_point_diff': 'home_rolling_point_diff'})
away_stats = team_games[['game_id', 'team', 'rolling_win_pct', 'rolling_point_diff']].rename(
    columns={'team': 'away_team', 'rolling_win_pct': 'away_rolling_win_pct', 'rolling_point_diff': 'away_rolling_point_diff'})

games = games.merge(home_stats, on=['game_id', 'home_team'], how='left')
games = games.merge(away_stats, on=['game_id', 'away_team'], how='left')

keep_cols = ['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 
             'home_win', 'div_game', 'home_rest', 'away_rest', 'spread_line', 
             'home_moneyline', 'away_moneyline', 'home_rolling_win_pct', 
             'away_rolling_win_pct', 'home_rolling_point_diff', 'away_rolling_point_diff']
games = games[keep_cols]

print(games.shape)
print(games[games['week'] == 1].head(10))  # check Week 1 games specifically — this is where carryover matters most

games.to_csv('data/processed_games.csv', index=False)
print("Saved to data/processed_games.csv")
