import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

games = pd.read_csv('data/processed_games.csv')

features = ['home_rest', 'away_rest', 'spread_line', 'div_game',
            'home_rolling_win_pct', 'away_rolling_win_pct',
            'home_rolling_point_diff', 'away_rolling_point_diff']

games = games.dropna(subset=features).reset_index(drop=True)

X = games[features]
y = games['home_win']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Keep the matching game info (teams, week) for the test set so we can show real examples
test_games = games.loc[X_test.index].copy()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)[:, 1]  # probability home team wins
accuracy = accuracy_score(y_test, predictions)

test_games['predicted_home_win'] = predictions
test_games['home_win_probability'] = probabilities
test_games['actual_home_win'] = y_test.values
test_games['correct'] = (test_games['predicted_home_win'] == test_games['actual_home_win'])

# ---- PLAIN-LANGUAGE SUMMARY ----
print("=" * 60)
print("NFL GAME PREDICTION MODEL — SUMMARY")
print("=" * 60)
print(f"\nThis model looked at {len(X_train)} past NFL games and learned")
print(f"patterns that predict who wins. It was then tested on")
print(f"{len(X_test)} new games it had never seen before.\n")
print(f"RESULT: It correctly predicted the winner in {accuracy:.1%} of games.")
print(f"(For comparison, always guessing 'home team wins' gets about 56%.)\n")

print("-" * 60)
print("A FEW EXAMPLE PREDICTIONS")
print("-" * 60)
sample = test_games.sample(5, random_state=1)
for _, row in sample.iterrows():
    home = row['home_team']
    away = row['away_team']
    prob = row['home_win_probability']
    predicted_winner = home if row['predicted_home_win'] == 1 else away
    actual_winner = home if row['actual_home_win'] == 1 else away
    result_icon = "correct" if row['correct'] else "incorrect"

    print(f"\n{away} @ {home} (Season {row['season']}, Week {row['week']})")
    print(f"  Model predicted: {predicted_winner} would win "
          f"({prob:.0%} confidence in {home})")
    print(f"  What actually happened: {actual_winner} won")
    print(f"  Prediction was: {result_icon}")

print("\n" + "=" * 60)
