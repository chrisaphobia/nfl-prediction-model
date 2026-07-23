import nfl_data_py as nfl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

years = [2022, 2023, 2024]
games = nfl.import_schedules(years)
games = games.dropna(subset=['home_score', 'away_score'])
games['home_win'] = (games['home_score'] > games['home_score'] * 0 + games['away_score']).astype(int)

features = ['home_rest', 'away_rest', 'spread_line', 'div_game']
games = games.dropna(subset=features)

X = games[features]
y = games['home_win']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model accuracy: {accuracy:.2%}")
print(f"Number of test games: {len(y_test)}")
