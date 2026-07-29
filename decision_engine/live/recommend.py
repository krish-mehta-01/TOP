"""
recommend.py
------------
The one function the backend calls: takes the raw scorecard JSON the
frontend sends and returns the full DecisionEngine result.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.engine import DecisionEngine  # noqa: E402
from live.live_state import (  # noqa: E402
    balls_to_dataframe,
    compute_batting_state,
    compute_bowling_state,
    phase_from_over,
)
from live.model_runner import get_runner  # noqa: E402

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")
_STATS_PATH = os.path.join(_CONFIG_DIR, "model_stats.json")

_engine = None


def get_engine() -> DecisionEngine:
    global _engine
    if _engine is None:
        _engine = DecisionEngine(config_dir=_CONFIG_DIR, stats_path=_STATS_PATH)
    return _engine


def recommend(payload: dict) -> dict:
    """
    payload = {
      "role": "bowling" | "batting",
      "venue": str, "batting_team": str, "bowling_team": str, "innings": 1|2,
      "balls": [ {over, ball_in_over, batter, non_striker, bowler,
                  batter_runs, wides, noballs, byes, legbyes, is_wicket}, ... ],
      "bowler": str,               # required if role == "bowling"
      "match_state_overrides": {}  # optional: recent_wicket, containment_confidence,
                                    #   balls_faced_by_batter, required_run_rate,
                                    #   projected_run_rate - anything the rule checks need
                                    #   that isn't derivable from `balls` alone.
    }
    """
    role = payload["role"]
    df = balls_to_dataframe(payload)
    runner = get_runner()

    if role == "bowling":
        bowler = payload.get("bowler") or df.iloc[-1]["bowler"]
        state = compute_bowling_state(df, bowler)
        raw_outputs = runner.predict_bowling(state)
        over = state["over"]
        match_state = {
            "phase": phase_from_over(over),
            "overs_bowled_by_current_bowler": state["overs_bowled_by_current_bowler"],
            "recent_wicket": bool(df["is_wicket"].tail(6).sum() > 0),
        }
    elif role == "batting":
        state = compute_batting_state(df)
        raw_outputs = runner.predict_batting(state)
        over = int(state["over"])
        match_state = {
            "phase": phase_from_over(over),
            "balls_faced_by_batter": state["balls_faced_by_current_batter"],
            "recent_wicket": state["recent_wicket"],
        }
    else:
        raise ValueError(f"unknown role: {role}")

    match_state.update(payload.get("match_state_overrides", {}))
    # required_run_rate / projected_run_rate (used by the death-overs defend
    # rule) still need a target score to compute and aren't derivable from
    # the ball log alone in a first innings - pass them via
    # match_state_overrides once a target exists (2nd innings chases).

    engine = get_engine()
    result = engine.decide(role=role, raw_model_outputs=raw_outputs, match_state=match_state)
    result["raw_model_outputs"] = raw_outputs
    result["state"] = {k: v for k, v in state.items() if not hasattr(v, "__len__") or isinstance(v, str)}
    return result
