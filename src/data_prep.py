import nfl_data_py as nfl

years = [2022, 2023, 2024]
games = nfl.import_schedules(years)

games = games.dropna(subset=['home_score', 'away_score'])

games['home_win'] = (games['home_score'] > games['away_score']).astype(int)

keep_cols = ['season', 'week', 'home_team', 'away_team', 'home_score', 
             'away_score', 'home_win', 'div_game', 'home_rest', 'away_rest',
             'spread_line', 'home_moneyline', 'away_moneyline']
games = games[keep_cols]

print(games.shape)
print(games.head(10))
print(games['home_win'].value_counts())
