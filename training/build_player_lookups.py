"""
build_player_lookups.py
------------------------
Builds two JSON lookup artifacts consumed by the "live" model family
(train_bowling_live.py / train_batting_live.py) as player-identity features,
and directly by the decision engine at inference time:

  - decision_engine/config/player_profiles.json  (per-batter, per-bowler career stats)
  - decision_engine/config/matchup_lookup.json    (per batter-vs-bowler pair stats)

Built strictly from TRAINING-match rows (match_level_split on the full
dataset, same seed=42 split every other script in this project uses) so
none of these lookups leak information from the held-out test matches that
train_bowling_live.py / train_batting_live.py evaluate against.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (  # noqa: E402
    add_bowling_state_features,
    load_dataset,
    match_level_split,
    save_json,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "traning_dataset", "dataset.csv")
CONFIG_DIR = os.path.join(HERE, "..", "decision_engine", "config")

BATTER_MIN_BALLS = 10
BOWLER_MIN_BALLS = 12  # at least 2 overs
PAIR_MIN_BALLS = 6
PAIR_HIGH_CONFIDENCE_BALLS = 30


def _batter_side_stats(rows, group_cols):
    """Shared aggregation for both the batter profile and the matchup lookup:
    balls_faced, runs_scored, strike_rate, dismissals (wicket_player_out ==
    batter), dismissal_rate, boundary_pct, dot_pct."""
    grouped = rows.groupby(group_cols).agg(
        balls_faced=("batter_runs", "size"),
        runs_scored=("batter_runs", "sum"),
        dismissals=("dismissal_flag", "sum"),
        boundary_sum=("is_boundary", "sum"),
        dot_sum=("is_dot", "sum"),
    ).reset_index()

    grouped["strike_rate"] = 100 * grouped["runs_scored"] / grouped["balls_faced"]
    grouped["dismissal_rate"] = grouped["dismissals"] / grouped["balls_faced"]
    grouped["boundary_pct"] = grouped["boundary_sum"] / grouped["balls_faced"]
    grouped["dot_pct"] = grouped["dot_sum"] / grouped["balls_faced"]
    return grouped


def main():
    df = load_dataset(DATA_PATH)
    print(f"Loaded dataset: {len(df):,} rows across {df['match_id'].nunique():,} matches")

    train_mask, test_mask = match_level_split(df)
    train_df = df.loc[train_mask].copy()
    print(f"Train-only rows: {len(train_df):,} across {train_df['match_id'].nunique():,} matches "
          f"(held out test matches: {df.loc[test_mask, 'match_id'].nunique():,})\n")

    # ---------------------------------------------------------------
    # Shared flags (computed only if not already present)
    # ---------------------------------------------------------------
    if "is_boundary" not in train_df.columns:
        train_df["is_boundary"] = train_df["batter_runs"].isin([4, 6])
    if "is_dot" not in train_df.columns:
        train_df["is_dot"] = train_df["total_runs"] == 0
    if "legal_ball" not in train_df.columns:
        train_df["legal_ball"] = (train_df["wides"] == 0) & (train_df["noballs"] == 0)

    # A dismissal is attributed to a row's own "batter" (striker on strike for
    # that ball) only when wicket_player_out matches that same name - this
    # correctly excludes run-outs of the NON-striker from that row (per spec).
    train_df["dismissal_flag"] = (train_df["wicket_player_out"] == train_df["batter"]).astype(int)

    # ---------------------------------------------------------------
    # Batter profile lookup
    # ---------------------------------------------------------------
    batter_group = _batter_side_stats(train_df, "batter")
    qualifying_batters = batter_group[batter_group["balls_faced"] >= BATTER_MIN_BALLS].copy()

    batters_lookup = {}
    for _, row in qualifying_batters.iterrows():
        batters_lookup[row["batter"]] = {
            "balls_faced": int(row["balls_faced"]),
            "runs_scored": int(row["runs_scored"]),
            "strike_rate": round(float(row["strike_rate"]), 3),
            "dismissals": int(row["dismissals"]),
            "dismissal_rate": round(float(row["dismissal_rate"]), 4),
            "boundary_pct": round(float(row["boundary_pct"]), 4),
            "dot_pct": round(float(row["dot_pct"]), 4),
        }

    batter_fallback = {
        "balls_faced": round(float(qualifying_batters["balls_faced"].mean()), 4),
        "runs_scored": round(float(qualifying_batters["runs_scored"].mean()), 4),
        "strike_rate": round(float(qualifying_batters["strike_rate"].mean()), 4),
        "dismissals": round(float(qualifying_batters["dismissals"].mean()), 4),
        "dismissal_rate": round(float(qualifying_batters["dismissal_rate"].mean()), 4),
        "boundary_pct": round(float(qualifying_batters["boundary_pct"].mean()), 4),
        "dot_pct": round(float(qualifying_batters["dot_pct"].mean()), 4),
    }

    # ---------------------------------------------------------------
    # Bowler profile lookup - replicates train_bowling.py's train_w12()
    # aggregation exactly (career_economy, dot_ball_percentage,
    # boundary_percentage, wickets_per_over), computed against the
    # train-only full dataframe, with a min total_balls >= 12 (2 overs)
    # inclusion threshold and its own global-mean fallback.
    # ---------------------------------------------------------------
    full_bowl = add_bowling_state_features(train_df)
    bowler_summary = (
        full_bowl.groupby("bowler")
        .agg(matches=("match_id", "nunique"), total_runs=("runs_conceded", "sum"),
             total_balls=("legal_delivery", "sum"), total_wickets=("is_wicket", "sum"),
             total_dot_balls=("is_dot_ball", "sum"), total_boundaries=("is_boundary", "sum"))
        .reset_index()
    )
    bowler_summary = bowler_summary[bowler_summary["total_balls"] >= BOWLER_MIN_BALLS].copy()

    bowler_summary["career_economy"] = np.where(
        bowler_summary["total_balls"] > 0, (bowler_summary["total_runs"] * 6) / bowler_summary["total_balls"], np.nan)
    bowler_summary["dot_ball_percentage"] = np.where(
        bowler_summary["total_balls"] > 0, bowler_summary["total_dot_balls"] / bowler_summary["total_balls"], np.nan)
    bowler_summary["boundary_percentage"] = np.where(
        bowler_summary["total_balls"] > 0, bowler_summary["total_boundaries"] / bowler_summary["total_balls"], np.nan)
    bowler_summary["wickets_per_over"] = np.where(
        bowler_summary["total_balls"] > 0, bowler_summary["total_wickets"] / (bowler_summary["total_balls"] / 6), np.nan)

    bowlers_lookup = {
        row["bowler"]: {
            "career_economy": round(row["career_economy"], 3),
            "dot_ball_percentage": round(row["dot_ball_percentage"], 4),
            "boundary_percentage": round(row["boundary_percentage"], 4),
            "wickets_per_over": round(row["wickets_per_over"], 3),
        }
        for _, row in bowler_summary.iterrows()
    }
    bowler_fallback = {
        "career_economy": round(float(bowler_summary["career_economy"].mean()), 3),
        "dot_ball_percentage": round(float(bowler_summary["dot_ball_percentage"].mean()), 4),
        "boundary_percentage": round(float(bowler_summary["boundary_percentage"].mean()), 4),
        "wickets_per_over": round(float(bowler_summary["wickets_per_over"].mean()), 3),
    }

    player_profiles = {
        "batters": batters_lookup,
        "bowlers": bowlers_lookup,
        "batter_fallback": batter_fallback,
        "bowler_fallback": bowler_fallback,
    }
    save_json(player_profiles, os.path.join(CONFIG_DIR, "player_profiles.json"))

    # ---------------------------------------------------------------
    # Matchup lookup: (batter, bowler) pairs
    # ---------------------------------------------------------------
    raw_pair_count = train_df.groupby(["batter", "bowler"]).ngroups

    pair_group = _batter_side_stats(train_df, ["batter", "bowler"])
    pair_group = pair_group[pair_group["balls_faced"] >= PAIR_MIN_BALLS].copy()
    pair_group["confidence"] = np.where(
        pair_group["balls_faced"] >= PAIR_HIGH_CONFIDENCE_BALLS, "high", "low")

    pairs_lookup = {}
    for _, row in pair_group.iterrows():
        key = f"{row['batter']}||{row['bowler']}"
        pairs_lookup[key] = {
            "balls_faced": int(row["balls_faced"]),
            "runs_scored": int(row["runs_scored"]),
            "strike_rate": round(float(row["strike_rate"]), 3),
            "dismissals": int(row["dismissals"]),
            "dismissal_rate": round(float(row["dismissal_rate"]), 4),
            "boundary_pct": round(float(row["boundary_pct"]), 4),
            "dot_pct": round(float(row["dot_pct"]), 4),
            "confidence": row["confidence"],
        }

    matchup_lookup = {"pairs": pairs_lookup, "min_balls_for_signal": PAIR_MIN_BALLS}
    save_json(matchup_lookup, os.path.join(CONFIG_DIR, "matchup_lookup.json"))

    # ---------------------------------------------------------------
    # Coverage summary
    # ---------------------------------------------------------------
    n_high = sum(1 for v in pairs_lookup.values() if v["confidence"] == "high")
    n_low = sum(1 for v in pairs_lookup.values() if v["confidence"] == "low")

    print("=" * 70)
    print("Player lookup coverage summary")
    print("=" * 70)
    print(f"Batters in lookup (balls_faced >= {BATTER_MIN_BALLS}):    {len(batters_lookup):,}")
    print(f"Bowlers in lookup (total_balls >= {BOWLER_MIN_BALLS}):    {len(bowlers_lookup):,}")
    print(f"Matchup pairs kept (balls_faced >= {PAIR_MIN_BALLS}):     {len(pairs_lookup):,}")
    print(f"  high confidence (>= {PAIR_HIGH_CONFIDENCE_BALLS} balls): {n_high:,}")
    print(f"  low confidence  ({PAIR_MIN_BALLS}-{PAIR_HIGH_CONFIDENCE_BALLS - 1} balls): {n_low:,}")
    print(f"Distinct (batter,bowler) pairs in raw train data (pre-filter): {raw_pair_count:,}")
    print(f"\nWrote {os.path.join(CONFIG_DIR, 'player_profiles.json')}")
    print(f"Wrote {os.path.join(CONFIG_DIR, 'matchup_lookup.json')}")


if __name__ == "__main__":
    main()
