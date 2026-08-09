# TOP — Tactical Optimization During Time-Out Period

Real-time decision support for cricket coaches: 28 retrained ML models feed a
unified decision engine that recommends one tactical action (bowling or batting)
from the current match state, with a live web console to drive it off a
ball-by-ball scorecard.

Read **PROJECT_REPORT.md** first — it documents every bug found in the original
system (data leakage, missing train/test splits, a broken model, two duplicate
models, an arbitrary confidence score) and exactly what changed, with honest
before/after numbers. Some models are genuinely weak; that's disclosed, not hidden.

## Project layout

```
data/                    dataset.csv (1,212 IPL matches, ~288k deliveries)
training/                common.py + train_bowling.py + train_batting.py + build_reliability.py
models/                  retrained .pkl models + encoding manifests (bowling/, batting/)
artifacts/output_data/   per-model metrics, predictions, training summaries
decision_engine/
  config/                model_config.json, actions.json, model_stats.json, reliability.json
  core/                  normalizer, weighter, aggregator, rule_validator, text_generator, engine
  live/                  live_state.py, model_runner.py, recommend.py — turns a live
                         scorecard into model features into a recommendation
backend/                 FastAPI app (app.py) exposing POST /api/recommend
frontend/                Vite + React live console (src/App.jsx)
PROJECT_REPORT.md        full list of fixes with before/after evidence
```

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
The right-hand panel shows the chosen action, a confidence score, a signal
readout for every candidate action (green = supports, red = against, the axis
is the neutral point), and which models contributed most to the winning action.

## Retraining from scratch

```bash
cd training
python3 train_bowling.py     # retrains W1-W15
python3 train_batting.py     # retrains B1-B15 (minus B4, which never existed, and B7)
python3 build_reliability.py # rebuilds reliability.json from the fresh metrics
```

This regenerates everything under `models/` and `artifacts/output_data/`. If you
change any model's feature list, update the matching feature order in
`decision_engine/live/model_runner.py` too — the live pipeline hardcodes each
model's exact training-time column order so a live prediction matches what the
model actually learned.

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
not hidden.
