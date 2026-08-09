# TOP — Tactical Optimization During Time-Out Period

Real-time decision support for cricket coaches: a suite of retrained ML models
feeds a unified decision engine that recommends one tactical action (bowling or
batting) from the current match state, ranked alongside genuine alternatives a
coach could choose instead, with a live web console to drive it off a
ball-by-ball scorecard.

Read **PROJECT_REPORT.md** first — it documents every bug found in the original
system (data leakage, missing train/test splits, a broken model, two duplicate
models, an arbitrary confidence score) and exactly what changed, with honest
before/after numbers. Some models are genuinely weak; that's disclosed, not
hidden — the same standard applies to every model suite below.

## Project layout

```
data/
  traning_dataset/dataset.csv   the real dataset: 1,212 T20 matches, ~288k
                                 deliveries, 2008-2026 (directory name has a
                                 known typo, kept as-is - see note below)
  output_data/                  per-model metrics/predictions/figures for the
                                 original model suite (models/bowling, models/batting)
  output_data_ballbyball/       same, for the ball-by-ball-retrained bowling
                                 suite (models/bowling_ballbyball) - see paper2/

training/
  common.py                     shared feature engineering + match-level split,
                                 used by every training script below
  train_bowling.py / train_batting.py
                                 the ORIGINAL suite: bowling trained on 4
                                 strategic-timeout snapshots per innings: batting
                                 mostly full-dataset already. Backs paper/.
  train_bowling_ballbyball.py   bowling retrained on every ball instead of the
                                 4 snapshots. Backs paper2/'s ablation.
  build_player_lookups.py, train_bowling_live.py, train_batting_live.py,
  verify_live.py, build_reliability_live.py
                                 in-progress full-history + player/matchup-aware
                                 retraining, intended to become the live app's
                                 default suite once complete (not yet wired in)
  build_reliability*.py         reliability.json builders, one per suite above

models/
  bowling/, batting/            original suite's .pkl models + encoding manifests
  bowling_ballbyball/           ball-by-ball-retrained bowling models

decision_engine/
  config/                       model_config.json, actions.json, model_stats.json,
                                 reliability.json (+ *_ballbyball variants),
                                 bowler_types.json (pace/spin for every roster
                                 bowler - see "Known limitations" below),
                                 player_profiles.json, matchup_lookup.json
                                 (train-only career/matchup stats, feed the
                                 in-progress live-suite retraining)
  core/                         normalizer, weighter, aggregator, rule_validator,
                                 text_generator, engine
  live/                         live_state.py, model_runner.py, recommend.py -
                                 turns a live scorecard into model features into
                                 a recommendation

backend/                        FastAPI app (app.py) exposing POST /api/recommend
frontend/                       Vite + React live console (src/App.jsx), team
                                 rosters in src/data/roster.json

paper/                          "TOP" - the original, strategic-timeout-scoped
                                 research paper (IEEE-style, PDF + source),
                                 built from training/train_bowling.py's and
                                 train_batting.py's real, verified output
paper2/                         "BOLT" - a standalone companion paper evaluating
                                 continuous, ball-by-ball tactical recommendations
                                 (does not reference paper/ - reads as fully
                                 independent work), built from
                                 train_bowling_ballbyball.py's real output

PROJECT_REPORT.md               full list of original-suite fixes with
                                 before/after evidence
```

**Which model suite is actually live?** `decision_engine/live/model_runner.py`
defaults to `models/bowling/` + `models/batting/` (the original suite). Set
`TOP_BOWLING_MODEL_DIR=bowling_ballbyball` to serve the ball-by-ball suite's
bowling models instead (batting is unaffected either way - it isn't retrained
between suites). The full-history, player/matchup-aware suite under
`training/*_live.py` is training data and scripts only for now - it is not yet
wired into the live decision engine.

**Known naming typo:** `data/traning_dataset/` should read `training_dataset` -
left uncorrected because `paper/build_b7_matchup.py` (part of the already
-verified, already-built research paper) reads from this exact path, and that
paper's build tooling is kept byte-identical to what its reported figures were
generated from.

## Running it

Requires Python 3.10+.

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Then open the URL Vite prints (typically `http://localhost:5173`). It talks
to `http://localhost:8000` — edit the `API_BASE` constant near the top of
`frontend/src/App.jsx` if you run the backend somewhere else.

Toggle **Bowling** / **Batting**, log the innings so far ball by ball (or click
**Load sample innings** to try it immediately), and click **Get Recommendation**.
The right-hand panel shows the chosen action, a confidence score, a
"Coach's Options" panel with the top pick plus genuine ranked alternatives (each
labeled with the model driving it), a full signal readout for every candidate
action (green = supports, red = against, the axis is the neutral point), and
which models contributed most to the winning action.

## Retraining from scratch

```bash
cd training
python3 train_bowling.py     # retrains W1-W15 (original, timeout-snapshot suite)
python3 train_batting.py     # retrains B1-B15 (minus B4, which never existed, and B7)
python3 build_reliability.py # rebuilds reliability.json from the fresh metrics
```

This regenerates everything under `models/bowling/`, `models/batting/`, and
`data/output_data/`. If you change any model's feature list, update the
matching feature order in `decision_engine/live/model_runner.py` too — the
live pipeline hardcodes each model's exact training-time column order so a
live prediction matches what the model actually learned.

## API

`POST /api/recommend`

```json
{
  "role": "bowling",
  "venue": "M Chinnaswamy Stadium, Bengaluru",
  "batting_team": "RCB", "bowling_team": "MI", "innings": 1,
  "bowler": "JJ Bumrah",
  "balls": [
    {"over": 0, "ball_in_over": 1, "batter": "V Kohli", "non_striker": "F du Plessis",
     "bowler": "JJ Bumrah", "batter_runs": 1, "wides": 0, "noballs": 0,
     "byes": 0, "legbyes": 0, "is_wicket": 0}
  ],
  "match_state_overrides": {}
}
```

`balls` is every delivery bowled so far in the current innings, in order — the
natural shape of a scorecard feed. `match_state_overrides` lets you pass
`required_run_rate` / `projected_run_rate` for the death-overs "don't defend
when miles behind the rate" rule, since that needs a chase target this endpoint
can't infer from the ball log alone.

Returns the chosen action, the full ranking, which (if any) higher-ranked
actions were blocked by a hard rule and why, a per-model contribution
breakdown, and the raw scalar every model produced.

## Known limitations (see PROJECT_REPORT.md §6 for detail)

`dataset.csv` has no bowler-type, delivery-type, or batter-handedness field, so
a handful of models (W7, W8, W9, W13, B9) can only approximate what their names
claim — this is disclosed per-model via a `note` field in `model_config.json`,
not hidden. Bowler pace/spin classification for the rule-validation layer
(which delivery variations a given bowler can legitimately be told to bowl)
comes from `decision_engine/config/bowler_types.json`, built from real,
publicly documented playing records for every bowler in
`frontend/src/data/roster.json` — a small number of minor/associate players
with no confidently verifiable record are listed in that file's
`_uncertain_defaulted_to_pace` key rather than guessed.
