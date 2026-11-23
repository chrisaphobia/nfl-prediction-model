# This project in plain English

A no-jargon guide to what this model does, what each number means, and how to read the results. All the numbers come from [`results/metrics.md`](results/metrics.md) and the [2026 scorecard](picks/2026/scorecard.md).

## What the model does

For every NFL game, the model gives a percentage: the chance that the home team wins. If it says 70%, it thinks the home team wins about 7 times out of 10 in games like this one. Its **pick** is whichever team it gives more than 50%.

## Where the data comes from

All the data comes from **nflreadpy**, a free Python library that downloads official NFL schedules, final scores, betting lines and every single play since 1999. It replaced an older library called nfl_data_py, which is no longer kept up to date. This project uses the 2015 to 2025 seasons to learn and test, and it uses 2026 for live picks.

## What the model looks at

The model only uses things that are known **before** the game starts. Using anything from after kickoff would be cheating, because the model would be "seeing the answer."

- **Recent form:** how often each team won in its last 3 games, and by how many points on average.
- **Last season:** in Week 1 there are no recent games yet, so it starts from how good each team was last year.
- **Elo rating:** a running strength score for every team (explained below).
- **Rest days:** how long each team has had off since its last game.
- **Situation:** whether it's a division rival, a playoff game, or a neutral-site game like London or the Super Bowl.
- **The betting spread (some versions only):** how many points Vegas expects the favorite to win by.

## How it was tested fairly

The model learned from the 2016 to 2024 seasons. Then it was tested on the **2025 season, which it never saw** while learning. It's like studying old exams and then taking a new one.

The original version of this project shuffled all the games together before testing. That meant it could learn from late-2025 games and then be "tested" on early-2024 games, like having next week's answers while taking this week's test. That made its score look better than it really was. Now the test season always comes after everything the model learned from.

**Walk-forward testing** repeats this for every season from 2018 to 2025. To predict 2019, it learns only from 2016 to 2018. To predict 2020, it learns from 2016 to 2019, and so on. That shows whether the model is good every year or just got lucky in one year.

## The two scores

### Accuracy: did it pick the winner?

This is the share of games where the team it picked actually won. **66.5% means about 2 right out of every 3.** That sounds low, but football is very random. Even Vegas only gets around 66%.

### Brier score: were the percentages any good?

Accuracy only asks "right or wrong." Brier score also asks **how confident** it was. For each game, it takes the gap between the model's percentage and what happened, squares it, and averages those across all games.

- **Lower is better.** A perfect model would score 0.
- **Always saying 50/50 scores 0.25.** That's the "knows nothing" line.
- Saying 90% and losing costs a lot more than saying 55% and losing. So a model can't get a good Brier score by acting overconfident.

Example: the model says the home team has a 70% chance and the home team wins. The miss is 1 − 0.70 = 0.30, which squared is 0.09. If the home team loses, the miss is 0.70, which squared is 0.49. Average that over every game and you get the Brier score.

**Log loss** (also in the results tables) is a stricter cousin of Brier score. It punishes confident wrong answers even harder. Lower is better there too.

## What the model was compared against

A number only means something next to a simple comparison. These are the comparisons, which people call **benchmarks**:

| Comparison | What it does |
|---|---|
| **Always pick the home team** | The laziest possible strategy. Home teams win a bit over half the time. |
| **Betting favorite** | Pick whoever Vegas favors. Vegas is very hard to beat, so this is the real test. Its percentage comes from the betting odds, with the bookmaker's cut removed. |
| **Elo rating** | A homemade team-strength score (see below). |

### What is Elo?

Elo is a rating system that started in chess. Every team starts at 1500. After each game, the winner takes points from the loser. Beating a strong team or winning big earns more points, and beating a weak team earns fewer. The home team gets a small bonus. Between seasons, every team drifts a third of the way back toward average, because rosters change. The gap between two teams' ratings turns into a win chance. This project computes Elo from scratch in [`src/elo.py`](src/elo.py).

## The results, in plain English

On the 2025 season the model never saw:

| Version | Accuracy | Brier | What it means |
|---|---:|---:|---|
| Always pick home team | 53.5% | 0.249 | Barely better than a coin flip. |
| **Betting favorite** | **66.2%** | **0.211** | The bar to beat. |
| Elo | 64.4% | 0.223 | A decent homemade rating. |
| **Model without the betting line** | **66.5%** | **0.220** | **Matched Vegas on picks using only football data.** |
| Model with the betting spread | 66.2% | 0.211 | It mostly just copied Vegas. |
| Model + EPA stats | 61.3% | 0.226 | Extra stats made it worse. |
| Gradient boosting + EPA | 64.4% | 0.224 | A fancier model didn't help either. |

Over all eight test seasons (2018 to 2025), the model without the betting line averages **64.2%** against **66.4%** for the favorite. So over many years, Vegas still leads by about 2 points. That's normal, and it's worth saying honestly.

### Why did the version with the spread just copy Vegas?

The betting spread already contains almost everything the other stats know, plus things they don't, like injuries and quarterback changes. Given the spread, the model leans on it and mostly ignores everything else. That's why the more interesting version is the one **without** the spread: it shows what the model figures out on its own.

### What is EPA, and why didn't it help?

**EPA (Expected Points Added)** measures how much each play helped a team's chances of scoring. For example, a 15-yard gain on 3rd-and-10 has a high EPA, and a sack has a negative one. Averaged over a season, it's one of the best measures of how good an offense or defense really is. **Success rate** is the share of plays that count as "successful" by that measure.

It didn't help here because Elo and recent point differential already capture most of the same information. EPA and Elo move together very closely, with a correlation of 0.88 on a scale where 1 means identical. Adding nine overlapping numbers to only about 2,500 games of training data mostly added noise.

### What is gradient boosting, and why didn't it help?

**Logistic regression** (the main model) adds up each factor with a weight, like "rest days count this much, Elo counts this much." It's simple and stable.

**Gradient boosting** builds hundreds of small decision trees ("if Elo gap > 50 and rest > 7 days, then…"), each one fixing the mistakes of the ones before it. It can find more complicated patterns, but it needs a lot of data to do that. With only about 2,500 games, it started memorizing quirks instead of learning real patterns. People call that **overfitting**.

Trying a stronger model and showing honestly that it didn't help is a real result. It shows the testing is fair and not tuned to make the numbers look good.

## Live 2026 picks and the scorecard

- **When picks are made:** a game gets picked as soon as both teams' previous games are final. That way the model uses every result available, and the pick is always made before kickoff.
- **Picks can't be changed:** every pick is saved in [`picks/2026/picks.csv`](picks/2026/picks.csv) and never edited. GitHub's history shows the exact time each pick was saved, so anyone can check it was made before the game.
- **The scorecard:** [`picks/2026/scorecard.md`](picks/2026/scorecard.md) grades every finished game. It shows the record, accuracy and Brier score for the model, the betting favorite and Elo, both week by week and for the season.
- **Which model picks:** the version **without** the betting line, so the picks are its own opinion rather than a copy of Vegas.
- **How often it updates:** an automatic job runs every morning at 7:47 AM Eastern, all the way through the Super Bowl.

## The one-sentence summary

> On a season it never saw, the model picked 66.5% of NFL winners (Brier score 0.220) using only football data, matching the betting favorite's 66.2%, and it beat a homemade Elo rating (64.4%).

## Words to know

| Word | Meaning |
|---|---|
| **Feature** | One piece of information the model uses, like rest days or Elo. |
| **Training** | The model studying past games to learn how much each feature matters. |
| **Held-out / test season** | A season kept hidden during training and used only to grade the model. |
| **Data leakage** | When information from the future sneaks into training, which makes results look better than they are. |
| **Spread** | How many points Vegas expects the favorite to win by. |
| **Moneyline** | Betting odds on who wins outright. They can be turned into a win percentage. |
| **Benchmark** | A simple comparison that tells you whether a model's score is actually good. |
| **Overfitting** | When a model memorizes the training games instead of learning patterns that work on new games. |
