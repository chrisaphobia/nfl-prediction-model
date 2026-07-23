# NFL Game Outcome Prediction Model

A logistic regression model that predicts NFL game winners using historical team performance data, built as a hands-on project to apply machine learning fundamentals.

## Overview

This project pulls real NFL game data (2022–2024 seasons), engineers a small set of predictive features, and trains a classification model to predict whether the home team will win. The goal was to build a complete, working ML pipeline end-to-end — from raw data to a trained, evaluated model — while learning core concepts in data preparation, feature selection, and model evaluation.

## Results

- **Model accuracy: 72.51%** on unseen test games
- **Baseline comparison:** simply guessing "home team always wins" would only achieve ~56% accuracy in this dataset, so the model adds meaningful predictive signal
- Tested on a held-out set of 171 games the model never saw during training

## Data

Data was pulled using the [`nfl_data_py`](https://pypi.org/project/nfl-data-py/) package, covering 854 games across the 2022–2024 NFL seasons.

**Features used:**
- `home_rest`, `away_rest` — days of rest before the game
- `spread_line` — the Vegas point spread
- `div_game` — whether it was a divisional matchup

**Target:**
- `home_win` — 1 if the home team won, 0 if they lost

## Approach

1. **Data collection** (`src/data_prep.py`) — pulls schedule/game data, drops unplayed games, creates the win/loss target label, and selects relevant columns
2. **Model training** (`src/model.py`) — splits data into training and test sets, trains a logistic regression classifier, and evaluates accuracy on unseen games

## How to Run

```bash
pip install -r requirements.txt
python3 src/data_prep.py   # view the cleaned dataset
python3 src/model.py       # train and evaluate the model
```

## Tech Stack

- Python
- pandas / NumPy — data handling
- scikit-learn — model training and evaluation
- nfl_data_py — data source

## Next Steps

- Add more features (team win percentage, recent scoring trends, turnover differential)
- Compare logistic regression against other models (random forest, gradient boosting)
- Build a script to predict upcoming week's games in real time

## Author

Christopher Gregory Jr. — [GitHub](https://github.com/chrisaphobia) | [LinkedIn](https://www.linkedin.com/in/chrisgregoryjr/)
