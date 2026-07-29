"""
live_state.py
--------------
Turns a live scorecard (a list of ball events the frontend is maintaining)
into the same "current state" fields the training pipeline computed from
dataset.csv - so a live prediction uses EXACTLY the same feature
definitions a model was trained on, not an approximation of them.

Input contract (what the frontend sends):
{
  "venue": str, "batting_team": str, "bowling_team": str, "innings": 1|2,
  "balls": [
     {"over": int, "ball_in_over": int, "batter": str, "non_striker": str,
      "bowler": str, "batter_runs": int, "wides": int, "noballs": int,
      "byes": int, "legbyes": int, "is_wicket": 0|1},
     ...
  ]
}
`balls` is every legal AND illegal delivery bowled so far in the CURRENT
innings, in order. That's the natural shape of a live scorecard feed - the
frontend appends one entry per ball as the innings happens.
"""

import numpy as np
import pandas as pd

PHASE_COLS = ["is_powerplay", "is_middle_overs", "is_death_overs"]


def _phase_flags(over: int) -> dict:
    return {
        "is_powerplay": int(over < 6),
        "is_middle_overs": int(6 <= over < 15),
        "is_death_overs": int(over >= 15),
    }


def balls_to_dataframe(context: dict) -> pd.DataFrame:
    balls = context.get("balls", [])
    if not balls:
        raise ValueError("balls list is empty - need at least one delivery to build any features")

    rows = []
    for b in balls:
        wides = int(b.get("wides", 0))
        noballs = int(b.get("noballs", 0))
        byes = int(b.get("byes", 0))
        legbyes = int(b.get("legbyes", 0))
        batter_runs = int(b.get("batter_runs", 0))
        total_runs = batter_runs + wides + noballs + byes + legbyes
        over = int(b["over"])
        row = {
            "match_id": "live", "innings": context.get("innings", 1),
            "over": over, "ball_in_over": int(b["ball_in_over"]),
            "batter": b.get("batter", "Unknown"), "non_striker": b.get("non_striker", "Unknown"),
            "bowler": b.get("bowler", "Unknown"),
            "batter_runs": batter_runs, "wides": wides, "noballs": noballs,
            "byes": byes, "legbyes": legbyes, "total_runs": total_runs,
            "extra_runs": total_runs - batter_runs,
            "is_wicket": int(b.get("is_wicket", 0)),
            "venue": context.get("venue", "Unknown"),
            "batting_team": context.get("batting_team", "Unknown"),
            "bowling_team": context.get("bowling_team", "Unknown"),
        }
        row.update(_phase_flags(over))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["over", "ball_in_over"]).reset_index(drop=True)
    return df


def compute_batting_state(df: pd.DataFrame) -> dict:
    """Current-innings state as of the LAST ball in df. Mirrors the exact
    running-total definitions used across B1/B3/B5/B6/B9-B15 at training
    time (see training/train_batting.py)."""
    d = df.copy()
    d["current_score"] = d["total_runs"].cumsum()
    d["wickets_fallen"] = d["is_wicket"].cumsum()
    d["balls_bowled"] = np.arange(1, len(d) + 1)
    d["balls_remaining"] = 120 - d["balls_bowled"]
    d["current_run_rate"] = d["current_score"] / (d["balls_bowled"] / 6)
    d["wickets_remaining"] = 10 - d["wickets_fallen"]

    d["is_boundary"] = d["batter_runs"].isin([4, 6]).astype(int)
    d["is_dot_ball"] = (d["total_runs"] == 0).astype(int)
    d["previous_runs"] = d["total_runs"].shift(1).fillna(0)
    d["recent_runs"] = d["total_runs"].rolling(6, min_periods=1).sum()
    d["recent_wickets"] = d["is_wicket"].rolling(12, min_periods=1).sum()
    d["dot_streak"] = d["is_dot_ball"].rolling(3, min_periods=1).sum()
    d["boundary_momentum"] = d["is_boundary"].rolling(6, min_periods=1).sum()
    d["boundary_count"] = d["is_boundary"].rolling(12, min_periods=1).sum()
    d["pressure_index"] = d["balls_remaining"] / (d["wickets_remaining"] + 1)
    d["momentum"] = d["recent_runs"] - (d["recent_wickets"] * 6)
    d["is_boundary_prev"] = d["batter_runs"].shift(1).isin([4, 6]).astype(int).fillna(0)

    # runs_last_6_balls (B1/B2) is the same as recent_runs but kept as its own
    # name to match those models' original column.
    d["runs_last_6_balls"] = d["recent_runs"]
    d["wicket_count"] = d["wickets_fallen"]
    d["wicket_density"] = d["wicket_count"] / (d["balls_bowled"] + 1)

    # partnership (B6): current unbroken partnership between the two batters
    # on the LAST ball's batter/non_striker pairing.
    last = d.iloc[-1]
    pid = "-".join(sorted([str(last["batter"]), str(last["non_striker"])]))
    d["_pid"] = d.apply(lambda r: "-".join(sorted([str(r["batter"]), str(r["non_striker"])])), axis=1)
    # partnership_runs = runs scored since the last wicket that broke the CURRENT pairing
    same_pid_from_end = (d["_pid"] == pid)[::-1]
    run_length = same_pid_from_end.cumprod().sum()  # contiguous match from the end
    partnership_runs = int(d["batter_runs"].tail(int(run_length)).sum())

    state = last.to_dict()
    state.update({
        "current_score": int(last["current_score"]),
        "wickets_fallen": int(last["wickets_fallen"]),
        "wickets_lost": int(last["wickets_fallen"]),
        "balls_bowled": int(last["balls_bowled"]),
        "balls_remaining": int(last["balls_remaining"]),
        "current_run_rate": float(last["current_run_rate"]),
        "wickets_remaining": int(last["wickets_remaining"]),
        "recent_runs": float(last["recent_runs"]),
        "recent_wickets": float(last["recent_wickets"]),
        "previous_runs": float(last["previous_runs"]),
        "dot_streak": float(last["dot_streak"]),
        "boundary_momentum": float(last["boundary_momentum"]),
        "boundary_count": float(last["boundary_count"]),
        "pressure_index": float(last["pressure_index"]),
        "momentum": float(last["momentum"]),
        "is_boundary_prev": int(last["is_boundary_prev"]),
        "runs_last_6_balls": float(last["runs_last_6_balls"]),
        "wicket_count": int(last["wicket_count"]),
        "wicket_density": float(last["wicket_density"]),
        "partnership_runs": partnership_runs,
        "balls_faced_by_current_batter": int((d["batter"] == last["batter"]).sum()),
        "recent_wicket": bool(d["is_wicket"].tail(6).sum() > 0),
    })
    return state


def compute_bowling_state(df: pd.DataFrame, bowler: str) -> dict:
    """Current-spell state for `bowler`, as of the last ball THEY bowled.
    Mirrors add_bowling_state_features() in training/common.py."""
    spell = df[df["bowler"] == bowler].copy()
    if spell.empty:
        raise ValueError(f"no balls found for bowler '{bowler}' in this innings yet")

    spell["runs_conceded"] = spell["batter_runs"] + spell["wides"] + spell["noballs"]
    spell["legal_delivery"] = ((spell["wides"] == 0) & (spell["noballs"] == 0)).astype(int)
    spell["is_dot_ball"] = (spell["runs_conceded"] == 0).astype(int)
    spell["is_boundary"] = (spell["batter_runs"] >= 4).astype(int)

    current_runs_conceded = int(spell["runs_conceded"].sum())
    current_balls_bowled = int(spell["legal_delivery"].sum())
    current_wickets = int(spell["is_wicket"].sum())
    current_dot_balls = int(spell["is_dot_ball"].sum())
    current_boundaries = int(spell["is_boundary"].sum())

    current_economy = (current_runs_conceded * 6 / current_balls_bowled) if current_balls_bowled > 0 else 0.0
    dot_ball_percentage = (current_dot_balls / current_balls_bowled) if current_balls_bowled > 0 else 0.0
    boundary_percentage = (current_boundaries / current_balls_bowled) if current_balls_bowled > 0 else 0.0

    last = spell.iloc[-1]
    over = int(last["over"])
    phase = "Powerplay" if over < 6 else ("Death" if over >= 15 else "Middle")

    return {
        "venue": last["venue"], "batting_team": last["batting_team"], "bowling_team": last["bowling_team"],
        "innings": last["innings"], "over": over,
        "current_runs_conceded": current_runs_conceded, "current_balls_bowled": current_balls_bowled,
        "current_wickets": current_wickets, "current_economy": current_economy,
        "dot_ball_percentage": dot_ball_percentage, "boundary_percentage": boundary_percentage,
        "match_phase": phase, "bowler": bowler,
        "overs_bowled_by_current_bowler": current_balls_bowled // 6,
    }


def phase_from_over(over: int) -> str:
    if over < 6:
        return "powerplay"
    if over < 15:
        return "middle"
    return "death"
