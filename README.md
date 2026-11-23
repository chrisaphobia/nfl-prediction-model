# NFL Game Outcome Prediction Model

Predicts the chance that the home team wins an NFL game, and scores those predictions honestly: every test season is one the model never trained on, and every result is compared against the betting favorite and a self-computed Elo rating.

## Live 2026 picks

The model makes a pick for every 2026 game before kickoff, including the playoffs, and grades itself against the betting favorite and Elo as results come in. See the running scorecard: [`picks/2026/scorecard.md`](picks/2026/scorecard.md). Picks are logged in [`picks/2026/picks.csv`](picks/2026/picks.csv) and never edited, so the git history shows each one was made in advance.

## Results

**Held-out 2025 season** (trained on 2016–2024, tested on all 284 games of 2025 including playoffs):

| Method | Accuracy | Brier | Log loss |
|---|---:|---:|---:|
| Always pick home team | 53.5% | 0.2490 | 0.6912 |
| Betting favorite | 66.2% | 0.2109 | 0.6070 |
| Elo (self-computed) | 64.4% | 0.2230 | 0.6366 |
| Logistic, no betting line | 66.5% | 0.2200 | 0.6300 |
| Logistic + EPA, no betting line | 61.3% | 0.2255 | 0.6412 |
| Gradient boosting + EPA, no betting line | 64.4% | 0.2243 | 0.6396 |
| Logistic, with spread | 66.2% | 0.2111 | 0.6069 |
| Logistic + EPA, with spread | 66.5% | 0.2146 | 0.6144 |
| Gradient boosting + EPA, with spread | 66.2% | 0.2112 | 0.6082 |

**Walk-forward 2018–2025** (each season predicted by a model trained only on earlier seasons, 2,219 games):

| Method | Accuracy | Brier | Log loss |
|---|---:|---:|---:|
| Always pick home team | 54.4% | 0.2486 | 0.6903 |
| Betting favorite | 66.4% | 0.2108 | 0.6089 |
| Elo (self-computed) | 63.5% | 0.2228 | 0.6366 |
| Logistic, no betting line | 64.2% | 0.2241 | 0.6393 |
| Logistic + EPA, no betting line | 63.9% | 0.2240 | 0.6397 |
| Gradient boosting + EPA, no betting line | 62.8% | 0.2278 | 0.6492 |
| Logistic, with spread | 66.4% | 0.2126 | 0.6129 |
| Logistic + EPA, with spread | 66.5% | 0.2140 | 0.6166 |
| Gradient boosting + EPA, with spread | 65.4% | 0.2163 | 0.6229 |

Per-season accuracy is in [`results/metrics.md`](results/metrics.md).

**In one line:** on the held-out 2025 season the model reaches 66.2% accuracy with a Brier score of 0.211, matching the betting favorite (66.2%, Brier 0.211). Without the betting line it reaches 66.5% (Brier 0.220), beating a self-computed Elo (64.4%, Brier 0.223).

**What this says:**
- The betting market is very hard to beat. With the spread as a feature, every model mostly learns to copy the line.
- The simple no-line logistic model is the most interesting result. Built only from schedule data and Elo, it lands close to the market and ahead of Elo on the 2025 holdout, though over all eight walk-forward seasons it trails the market by about two points of accuracy.
- Play-by-play EPA and success rate did **not** help. Elo and point differential already carry most of the same signal (EPA net difference and Elo difference correlate at 0.88), and adding nine overlapping features to about 2,500 training games adds noise.
- Gradient boosting did **not** beat logistic regression either. With this little data and mostly linear signal, the extra flexibility overfits. The feature sets and boosting settings were fixed before scoring, not tuned on the test seasons.

Brier score is the mean squared error of the win probability (lower is better). Always predicting 50% scores 0.25.

## How it works

1. **Data** (`src/data_prep.py`): schedules, scores, rest days and betting lines for 2015–2025 from [nflreadpy](https://github.com/nflverse/nflreadpy), which replaced the deprecated `nfl_data_py`. Relocated franchises (OAK→LV, SD→LAC, STL→LA) are merged so ratings follow the team.
2. **Features**, all known before kickoff:
   - 3-game rolling win % and point differential within a season, shifted so a game never sees its own result. Early-season gaps are filled with last season's averages.
   - Pre-game Elo difference (`src/elo.py`): K = 20, 48 points of home field (none at neutral sites), a margin-of-victory multiplier, and one-third regression to the mean between seasons.
   - Play-by-play efficiency from `nflreadpy.load_pbp`: offensive and defensive EPA/play and success rate on pass and run plays, averaged over a team's earlier games that season and blended with last season's averages (worth 4 games) while the sample is small.
   - Rest days, division game, neutral site, playoff flag and, in some variants, the closing spread.
3. **Models** (`src/model.py`): standardized logistic regression, and scikit-learn's `HistGradientBoostingClassifier` (shallow trees, slow learning rate, strong regularization).
4. **Evaluation** (`src/evaluate.py`): accuracy, Brier score and log loss against the benchmarks above. The favorite's probability is the no-vig moneyline. Ties are excluded.

## Run it

```bash
pip install -r requirements.txt
python src/data_prep.py   # downloads schedules and play-by-play, writes data/processed_games.csv
python src/evaluate.py    # writes results/metrics.md
python src/model.py       # plain-language summary and example predictions
python src/picks.py       # new picks for upcoming games + updated 2026 scorecard
```

## Next steps

- Weight EPA toward recent games and adjust it for opponent strength, instead of a flat season-to-date average.
- Add quarterback changes and injuries, which Elo and season averages miss but the betting line captures.
