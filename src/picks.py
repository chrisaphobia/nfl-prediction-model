"""Live weekly picks with a running scorecard.

Each run does two things:
1. Makes a pick for every upcoming game that is the next game for both teams
   (so every feature reflects all games played so far) and hasn't kicked off.
   Picks are appended to picks/<season>/picks.csv and never changed afterwards,
   so the git history proves each pick was made before kickoff.
2. Grades every pick whose game is final and rewrites picks/<season>/scorecard.md,
   comparing the model with the betting favorite (closing line) and Elo.

The pick model is the logistic regression *without* the betting line, trained on
2016 through the previous season, so its picks are its own opinion rather than a
copy of the market.

Run from the repo root:  python src/picks.py
"""
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import nflreadpy as nfl
import pandas as pd
from sklearn.metrics import brier_score_loss

from data_prep import FIRST_SEASON, build
from model import TEAM_FEATURES, TRAIN_START, fit

SEASON = int(os.environ.get('PICKS_SEASON', 2026))
OUT_DIR = f'picks/{SEASON}'
PICKS_PATH = f'{OUT_DIR}/picks.csv'
SCORECARD_PATH = f'{OUT_DIR}/scorecard.md'
EASTERN = ZoneInfo('America/New_York')
PICK_COLS = ['game_id', 'season', 'week', 'game_type', 'gameday', 'gametime', 'away_team', 'home_team',
             'model_home_prob', 'model_pick', 'elo_home_prob', 'elo_pick', 'picked_at']


def kickoff(row):
    time = row['gametime'] if isinstance(row['gametime'], str) else '13:00'
    return datetime.strptime(f"{row['gameday']:%Y-%m-%d} {time}", '%Y-%m-%d %H:%M').replace(tzinfo=EASTERN)


def pickable(season_games, now):
    """Unplayed games that are each team's next game and haven't started."""
    unplayed = season_games[season_games['home_score'].isna()].sort_values('gameday')
    next_game = {}
    for row in unplayed.itertuples():
        for team in (row.home_team, row.away_team):
            next_game.setdefault(team, row.game_id)
    ok = unplayed.apply(lambda r: next_game[r['home_team']] == r['game_id'] == next_game[r['away_team']]
                        and kickoff(r) > now, axis=1)
    return unplayed[ok] if len(unplayed) else unplayed


def load_picks():
    if os.path.exists(PICKS_PATH):
        return pd.read_csv(PICKS_PATH)
    return pd.DataFrame(columns=PICK_COLS)


def make_picks(games, picks, now):
    history = games[(games['season'] >= TRAIN_START) & (games['season'] < SEASON)
                    & games['home_score'].notna() & (games['home_score'] != games['away_score'])].copy()
    history['home_win'] = (history['home_score'] > history['away_score']).astype(int)
    model = fit(history, TEAM_FEATURES)

    new = pickable(games[games['season'] == SEASON], now)
    new = new[~new['game_id'].isin(picks['game_id'])].copy()
    if new.empty:
        return picks, new
    new['model_home_prob'] = model.predict_proba(new[TEAM_FEATURES])[:, 1].round(4)
    new['model_pick'] = new['home_team'].where(new['model_home_prob'] > 0.5, new['away_team'])
    new['elo_home_prob'] = new['elo_home_prob'].round(4)
    new['elo_pick'] = new['home_team'].where(new['elo_home_prob'] > 0.5, new['away_team'])
    new['picked_at'] = now.strftime('%Y-%m-%d %H:%M %Z')
    new['gameday'] = new['gameday'].dt.strftime('%Y-%m-%d')
    return pd.concat([picks, new[PICK_COLS]], ignore_index=True), new


def grade(games, picks):
    """Join picks to final scores and the closing betting line."""
    results = games[['game_id', 'home_score', 'away_score', 'market_home_prob']]
    g = picks.merge(results, on='game_id', how='left')
    g = g[g['home_score'].notna() & (g['home_score'] != g['away_score'])].copy()  # final, not a tie
    g['home_win'] = (g['home_score'] > g['away_score']).astype(int)
    g['winner'] = g['home_team'].where(g['home_win'] == 1, g['away_team'])
    g['fav_pick'] = g['home_team'].where(g['market_home_prob'] > 0.5, g['away_team'])
    for name in ['model', 'elo', 'fav']:
        g[f'{name}_correct'] = (g[f'{name}_pick'] == g['winner']).astype(int)
    return g


def summary_row(y, prob, correct):
    if len(y) == 0:
        return '0-0', '–', '–'
    wins = int(correct.sum())
    return f'{wins}-{len(y) - wins}', f'{wins / len(y):.1%}', f'{brier_score_loss(y, prob):.4f}'


def write_scorecard(picks, graded, games, now):
    lines = [f'# {SEASON} live picks scorecard', '',
             f'Updated {now:%A, %B %-d, %Y at %-I:%M %p} ET. Picks are made before kickoff and never edited '
             '(see the git history of `picks.csv`). The model is logistic regression without the betting line; '
             'the favorite uses the closing moneyline. Ties are not graded.', '']

    lines += ['## Season to date', '', '| Method | Record | Accuracy | Brier |', '|---|---:|---:|---:|']
    has_line = graded['market_home_prob'].notna()
    for label, name, prob, mask in [('Model', 'model', 'model_home_prob', None),
                                    ('Betting favorite', 'fav', 'market_home_prob', has_line),
                                    ('Elo', 'elo', 'elo_home_prob', None)]:
        d = graded if mask is None else graded[mask]
        rec, acc, brier = summary_row(d['home_win'], d[prob], d[f'{name}_correct'])
        lines.append(f'| {label} | {rec} | {acc} | {brier} |')

    if not graded.empty:
        lines += ['', '## By week', '', '| Week | Games | Model | Favorite | Elo |', '|---|---:|---:|---:|---:|']
        for week, d in graded.groupby('week'):
            lines.append(f"| {week} | {len(d)} | {d['model_correct'].sum()} | {d['fav_correct'].sum()} | {d['elo_correct'].sum()} |")

    pending = picks[~picks['game_id'].isin(graded['game_id'])]
    final_ids = set(games.loc[games['home_score'].notna(), 'game_id'])
    pending = pending[~pending['game_id'].isin(final_ids)]
    if not pending.empty:
        lines += ['', '## Upcoming picks', '', '| Week | Date | Game | Model pick | Model win prob. | Elo pick |',
                  '|---|---|---|---|---:|---|']
        for r in pending.sort_values(['gameday', 'gametime']).itertuples():
            prob = r.model_home_prob if r.model_pick == r.home_team else 1 - r.model_home_prob
            lines.append(f'| {r.week} | {r.gameday} | {r.away_team} @ {r.home_team} | **{r.model_pick}** | {prob:.0%} | {r.elo_pick} |')

    if not graded.empty:
        last_week = graded['week'].max()
        d = graded[graded['week'] == last_week]
        lines += ['', f'## Week {last_week} results', '', '| Game | Final | Model | Favorite | Elo |', '|---|---|---|---|---|']
        mark = {1: '✅', 0: '❌'}
        for r in d.itertuples():
            lines.append(f'| {r.away_team} @ {r.home_team} | {int(r.away_score)}–{int(r.home_score)} | '
                         f'{r.model_pick} {mark[r.model_correct]} | {r.fav_pick} {mark[r.fav_correct]} | {r.elo_pick} {mark[r.elo_correct]} |')

    with open(SCORECARD_PATH, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    now = datetime.now(EASTERN)
    games = build(FIRST_SEASON, SEASON)
    sched = nfl.load_schedules([SEASON]).to_pandas()[['game_id', 'gametime']]
    games = games.merge(sched, on='game_id', how='left')

    os.makedirs(OUT_DIR, exist_ok=True)
    picks, new = make_picks(games, load_picks(), now)
    picks.to_csv(PICKS_PATH, index=False)
    graded = grade(games, picks)
    write_scorecard(picks, graded, games, now)

    print(f'{len(new)} new picks, {len(picks)} total, {len(graded)} graded.')
    for r in new.itertuples():
        print(f'  Week {r.week}: {r.away_team} @ {r.home_team} -> {r.model_pick} ({r.model_home_prob:.0%} home)')


if __name__ == '__main__':
    main()
