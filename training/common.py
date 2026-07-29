"""
common.py
---------
Shared utilities for retraining every W/B model with one consistent,
leakage-free methodology:

  - one dataset loader
  - one definition of the "current state" features (bowling side / batting side)
  - one train/test split function (ALWAYS match-level, never row-level)
  - one set of metric + artifact writers

Every individual train_W*.py / train_B*.py script imports from here instead
of re-implementing its own version of these things. This is the fix for the
inconsistent-methodology problem in the original notebooks: some of them
grouped by match_id before splitting (correct), others called
train_test_split() directly on rows (incorrect - lets a single match leak
across train and test).
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

RANDOM_SEED = 42
TIMEOUT_OVERS = [5, 8, 12, 15]  # strategic timeout snapshots, ball 6 of that over


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df.sort_values(
        by=["match_id", "innings", "over", "ball_in_over"]
    ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Shared feature blocks
# ---------------------------------------------------------------------------

def add_bowling_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    The single feature template shared by every clean W-model
    (W1,W2,W3,W4,W6,W7,W8,W9,W11,W13,W15): running figures for the CURRENT
    bowler's spell, computed strictly from balls bowled SO FAR. None of these
    reference the outcome of a not-yet-known future ball, so they are safe
    to use as predictors for any "what happens next" target.
    """
    df = df.copy()
    df["runs_conceded"] = df["batter_runs"] + df["wides"] + df["noballs"]
    df["legal_delivery"] = ((df["wides"] == 0) & (df["noballs"] == 0)).astype(int)
    df["is_dot_ball"] = (df["runs_conceded"] == 0).astype(int)
    df["is_boundary"] = (df["batter_runs"] >= 4).astype(int)

    group_cols = ["match_id", "innings", "bowler"]
    df["current_runs_conceded"] = df.groupby(group_cols)["runs_conceded"].cumsum()
    df["current_balls_bowled"] = df.groupby(group_cols)["legal_delivery"].cumsum()
    df["current_wickets"] = df.groupby(group_cols)["is_wicket"].cumsum()
    df["current_dot_balls"] = df.groupby(group_cols)["is_dot_ball"].cumsum()
    df["current_boundaries"] = df.groupby(group_cols)["is_boundary"].cumsum()

    df["current_economy"] = np.where(
        df["current_balls_bowled"] > 0,
        (df["current_runs_conceded"] * 6) / df["current_balls_bowled"],
        0,
    )
    df["dot_ball_percentage"] = np.where(
        df["current_balls_bowled"] > 0,
        df["current_dot_balls"] / df["current_balls_bowled"],
        0,
    )
    df["boundary_percentage"] = np.where(
        df["current_balls_bowled"] > 0,
        df["current_boundaries"] / df["current_balls_bowled"],
        0,
    )
    df["match_phase"] = np.select(
        [df["is_powerplay"] == 1, df["is_middle_overs"] == 1, df["is_death_overs"] == 1],
        ["Powerplay", "Middle", "Death"],
        default="Middle",
    )
    return df


BOWLING_SNAPSHOT_FEATURES = [
    "venue", "batting_team", "bowling_team", "innings", "over",
    "current_runs_conceded", "current_balls_bowled", "current_wickets",
    "current_economy", "dot_ball_percentage", "boundary_percentage", "match_phase",
]
BOWLING_CATEGORICAL = ["venue", "batting_team", "bowling_team", "innings", "match_phase"]


def bowling_timeout_snapshot(df: pd.DataFrame, require_future_balls: bool = False) -> pd.DataFrame:
    """
    Restricts to the actual strategic-timeout moments (end of overs 5/8/12/15)
    used to train every W-model. If require_future_balls is set, also drops
    rows where the bowler has nothing left to bowl (used by models whose
    target describes the REST of the spell).
    """
    snap = df[(df["over"].isin(TIMEOUT_OVERS)) & (df["ball_in_over"] == 6)].copy()
    if require_future_balls:
        snap = snap[snap["future_balls"] > 0].copy()
    return snap


def add_future_spell_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    For a bowler's spell, adds final_runs/final_balls/final_boundaries (the
    spell's eventual totals) and future_* (what's left AFTER the current
    ball). future_* is what several targets (future_runs, future_wicket,
    death_over_accuracy, ...) are built from - it never includes anything
    already known at decision time.
    """
    df = df.copy()
    group_cols = ["match_id", "innings", "bowler"]
    summary = (
        df.groupby(group_cols)
        .agg(
            final_runs=("runs_conceded", "sum"),
            final_balls=("legal_delivery", "sum"),
            final_boundaries=("is_boundary", "sum"),
            final_wickets=("is_wicket", "sum"),
            final_dot_balls=("is_dot_ball", "sum"),
        )
        .reset_index()
    )
    df = df.merge(summary, on=group_cols, how="left")
    df["future_runs"] = df["final_runs"] - df["current_runs_conceded"]
    df["future_balls"] = df["final_balls"] - df["current_balls_bowled"]
    df["future_boundaries"] = df["final_boundaries"] - df["current_boundaries"]
    df["future_wickets"] = df["final_wickets"] - df["current_wickets"]
    df["future_dot_balls"] = df["final_dot_balls"] - df["current_dot_balls"]
    df["future_economy"] = np.where(
        df["future_balls"] > 0, (df["future_runs"] * 6) / df["future_balls"], np.nan
    )
    df["future_boundary_percentage"] = np.where(
        df["future_balls"] > 0, df["future_boundaries"] / df["future_balls"], np.nan
    )
    df["future_dot_ball_percentage"] = np.where(
        df["future_balls"] > 0, df["future_dot_balls"] / df["future_balls"], np.nan
    )
    df["final_dot_ball_percentage"] = np.where(
        df["final_balls"] > 0, df["final_dot_balls"] / df["final_balls"], np.nan
    )
    return df


def add_batting_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shared running-innings features for batting models: score, wickets,
    balls used/remaining, run rate - all computed up to AND INCLUDING the
    current ball, which is fine for features (never for target leakage,
    see NOTE in each model file about which of these should NOT double as
    a feature when they equal the target).
    """
    df = df.copy()
    grp = ["match_id", "innings"]
    df["total_balls"] = df["over"] * 6 + df["ball_in_over"]
    df["current_score"] = df.groupby(grp)["total_runs"].cumsum()
    df["wickets_lost"] = df.groupby(grp)["is_wicket"].cumsum()
    df["balls_bowled_innings"] = df.groupby(grp).cumcount() + 1
    df["balls_remaining"] = 120 - df["balls_bowled_innings"]
    df["wickets_remaining"] = 10 - df["wickets_lost"]
    df["current_run_rate"] = np.where(
        df["balls_bowled_innings"] > 0,
        df["current_score"] / (df["balls_bowled_innings"] / 6),
        0,
    )
    return df


# ---------------------------------------------------------------------------
# Splitting - ALWAYS match-level. This is the one function every model
# (bowling or batting) must use instead of sklearn's row-level train_test_split.
# ---------------------------------------------------------------------------

def match_level_split(df: pd.DataFrame, id_col: str = "match_id", test_size: float = 0.20, seed: int = RANDOM_SEED):
    unique_matches = df[id_col].unique().copy()
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_matches)
    split_idx = int((1 - test_size) * len(unique_matches))
    train_matches = set(unique_matches[:split_idx])
    test_matches = set(unique_matches[split_idx:])
    train_mask = df[id_col].isin(train_matches)
    test_mask = df[id_col].isin(test_matches)
    return train_mask, test_mask


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_pred, y_prob) -> dict:
    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["ROC_AUC"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["ROC_AUC"] = None
    return out


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def save_metrics(metrics: dict, path: str):
    pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())}).to_csv(
        path, index=False
    )


def save_feature_importance(features, importances, path: str):
    pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(
        "Importance", ascending=False
    ).to_csv(path, index=False)


def print_header(model_id: str, model_name: str):
    print("=" * 70)
    print(f"{model_id} : {model_name}")
    print("=" * 70)


def print_result(metrics: dict):
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:10s}: {v:.4f}")
        else:
            print(f"  {k:10s}: {v}")


# ---------------------------------------------------------------------------
# Categorical encoding - THREE consistent strategies, each saved to a JSON
# "encoding manifest" next to the model so the live feature pipeline can
# reproduce the exact same column layout at inference time.
# ---------------------------------------------------------------------------

def fit_native_categories(df: pd.DataFrame, cat_cols: list) -> dict:
    """For LightGBM / XGBoost native categorical support. Returns the fixed
    category vocabulary (from TRAINING data only) per column."""
    return {c: sorted(df[c].astype(str).unique().tolist()) for c in cat_cols}


def apply_native_categories(df: pd.DataFrame, cat_cols: list, vocab: dict) -> pd.DataFrame:
    df = df.copy()
    for c in cat_cols:
        cats = vocab[c]
        df[c] = pd.Categorical(df[c].astype(str), categories=cats)
    return df


def fit_onehot_schema(df: pd.DataFrame, cat_cols: list) -> list:
    """For LogisticRegression / RandomForest / GaussianNB (no native
    categorical support). Returns the fixed list of one-hot column names
    produced from TRAINING data - live rows are reindexed onto this exact
    list so the model always sees the same column order/width."""
    dummies = pd.get_dummies(df[cat_cols].astype(str), columns=cat_cols)
    return dummies.columns.tolist()


def apply_onehot_schema(df: pd.DataFrame, cat_cols: list, numeric_cols: list, schema: list) -> pd.DataFrame:
    dummies = pd.get_dummies(df[cat_cols].astype(str), columns=cat_cols)
    dummies = dummies.reindex(columns=schema, fill_value=0)
    return pd.concat([df[numeric_cols].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)


def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
