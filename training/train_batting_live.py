"""
train_batting_live.py
-----------------------
Retrains 13 of the 15 batting models (B1-B3, B5-B6, B8-B15) for the "live"
system - an isolated sibling of train_batting.py. train_batting.py,
models/batting/, and artifacts/output_data/ are never modified, run, or
read from here. B7 (Matchup Matrix) remains out of scope, same as in
train_batting.py.

Unlike train_bowling_live.py, row selection is UNCHANGED for every model -
B1,B2,B3,B5,B6,B9,B10,B12,B13 already train on the full dataframe in
train_batting.py, B8 keeps its is_powerplay==1 filter, B11 keeps its "last
ball of every over" .tail(1) grouping, B14/B15 keep their over>=15 filter.

What's new is player-identity features, split by algorithm family:
  - Group A (LightGBM / XGBoost(enable_categorical) / CatBoost multiclass):
    B1, B2, B3, B10, B11, B14, B15 get "batter"/"bowler" added as native
    categorical columns (B10/B11/B14/B15 previously had NO categorical
    features at all - fit_native_categories/apply_native_categories now
    used the same way B1 already used them; B15 mirrors W3's XGBoost
    enable_categorical pattern).
  - Group B (LogisticRegression / RandomForestRegressor / CatBoost binary /
    KNeighborsClassifier - one-hotting hundreds of player names is
    impractical): B5, B6, B9, B12, B13 instead get DERIVED NUMERIC columns
    looked up from decision_engine/config/player_profiles.json (batter
    always; B6 also non_striker; B9 also two bowler stats).
  - B8 gets neither - left exactly as in train_batting.py (only path/data
    changes), per spec.

Same algorithms, hyperparameters, target definitions, and match-level
(seed=42) split as train_batting.py. Outputs go to NEW, isolated paths:
  - models/batting_live/
  - data/output_data_live/batting/
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(__file__))
from common import (  # noqa: E402
    apply_native_categories,
    classification_metrics,
    ensure_dirs,
    fit_native_categories,
    load_dataset,
    match_level_split,
    print_header,
    print_result,
    regression_metrics,
    save_feature_importance,
    save_json,
    save_metrics,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "traning_dataset", "dataset.csv")
MODEL_ROOT = os.path.join(HERE, "..", "models", "batting_live")
OUT_ROOT = os.path.join(HERE, "..", "data", "output_data_live", "batting")
CONFIG_DIR = os.path.join(HERE, "..", "decision_engine", "config")

# Loaded once, module-level, before any per-model function runs.
with open(os.path.join(CONFIG_DIR, "player_profiles.json")) as _f:
    PLAYER_PROFILES = json.load(_f)

RESULTS = {}


def dirs_for(model_id):
    mod = os.path.join(MODEL_ROOT, model_id)
    out = os.path.join(OUT_ROOT, model_id)
    ensure_dirs(mod, out)
    return mod, out


def add_batter_profile_cols(df, col="batter", prefix="batter"):
    """Group B: 3 derived numeric columns looked up from PLAYER_PROFILES via
    the given column (batter, or non_striker for B6), with a global fallback."""
    df = df.copy()
    batters = PLAYER_PROFILES["batters"]
    fallback = PLAYER_PROFILES["batter_fallback"]
    df[f"{prefix}_career_strike_rate"] = df[col].map(lambda b: batters.get(b, fallback)["strike_rate"])
    df[f"{prefix}_career_dismissal_rate"] = df[col].map(lambda b: batters.get(b, fallback)["dismissal_rate"])
    df[f"{prefix}_career_boundary_pct"] = df[col].map(lambda b: batters.get(b, fallback)["boundary_pct"])
    return df


def add_bowler_profile_cols_partial(df, col="bowler"):
    """B9 only: bowler_career_economy + bowler_career_dot_pct (not the full 4-field set)."""
    df = df.copy()
    bowlers = PLAYER_PROFILES["bowlers"]
    fallback = PLAYER_PROFILES["bowler_fallback"]
    df["bowler_career_economy"] = df[col].map(lambda b: bowlers.get(b, fallback)["career_economy"])
    df["bowler_career_dot_pct"] = df[col].map(lambda b: bowlers.get(b, fallback)["dot_ball_percentage"])
    return df


# ---------------------------------------------------------------------------
# B1 : Run Projection (LightGBM Regressor) - Group A
# ---------------------------------------------------------------------------
def train_b1(df):
    mid, name = "B1", "Run Projection"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)

    df = df.copy()
    df["final_innings_total"] = df.groupby(["match_id", "innings"])["total_runs"].transform("sum")
    df["wicket_count"] = df.groupby(["match_id", "innings"])["is_wicket"].cumsum()
    df["total_balls"] = df["over"] * 6 + df["ball_in_over"]
    df["balls_remaining"] = 120 - df["total_balls"]
    df["cumulative_runs"] = df.groupby(["match_id", "innings"])["total_runs"].cumsum()
    df["current_run_rate"] = np.where(df["total_balls"] > 0, (df["cumulative_runs"] / df["total_balls"]) * 6, 0.0)
    df["runs_last_6_balls"] = df.groupby(["match_id", "innings"])["total_runs"].transform(
        lambda x: x.rolling(6, min_periods=1).sum())

    cat_cols = ["venue", "batting_team", "bowling_team", "innings", "batter", "bowler"]
    features = cat_cols + ["over", "balls_remaining", "wicket_count", "current_run_rate", "runs_last_6_balls"]

    train_mask, test_mask = match_level_split(df)
    vocab = fit_native_categories(df.loc[train_mask], cat_cols)
    X = apply_native_categories(df[features], cat_cols, vocab)
    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train = df.loc[train_mask, "final_innings_total"]
    y_test = df.loc[test_mask, "final_innings_total"]

    model = LGBMRegressor(n_estimators=250, learning_rate=0.05, num_leaves=63, random_state=42, n_jobs=-1, verbosity=-1)
    model.fit(X_train, y_train, categorical_feature=cat_cols)
    pred = model.predict(X_test)
    metrics = regression_metrics(y_test, pred)
    print_result(metrics)

    joblib.dump(model, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_json(vocab, os.path.join(mod_dir, f"{mid}_categories.json"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    save_feature_importance(features, model.feature_importances_, os.path.join(out_dir, f"{mid}_feature_importance.csv"))
    results = df.loc[test_mask, ["match_id", "innings", "over", "ball_in_over"]].copy()
    results["b1_actual_score"] = y_test.values
    results["b1_predicted_score"] = pred
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "batting", "label": name, "column": "b1_predicted_score",
                     "type": "regression", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics}


# ---------------------------------------------------------------------------
# B2 : Dismissal Risk (XGBoost Cox survival) - Group A
# ---------------------------------------------------------------------------
def train_b2(df):
    mid, name = "B2", "Dismissal Risk"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)

    df = df.copy()
    df["total_balls"] = df["over"] * 6 + df["ball_in_over"]
    df["balls_remaining"] = 120 - df["total_balls"]
    df["wicket_count"] = df.groupby(["match_id", "innings"])["is_wicket"].cumsum()
    df["cumulative_runs"] = df.groupby(["match_id", "innings"])["total_runs"].cumsum()
    df["current_run_rate"] = np.where(df["total_balls"] > 0, (df["cumulative_runs"] / df["total_balls"]) * 6, 0.0)
    df["runs_last_6_balls"] = df.groupby(["match_id", "innings"])["total_runs"].transform(
        lambda x: x.rolling(6, min_periods=1).sum())
    df["wicket_density"] = df["wicket_count"] / (df["total_balls"] + 1)

    df["wicket_ball_marker"] = np.where(df["is_wicket"] == 1, df["total_balls"], np.nan)
    df["next_wicket_ball"] = df.groupby(["match_id", "innings"])["wicket_ball_marker"].bfill()
    df["max_balls_innings"] = df.groupby(["match_id", "innings"])["total_balls"].transform("max")
    df["time_to_event"] = df["next_wicket_ball"] - df["total_balls"] + 1
    df["survival_target"] = np.where(
        df["next_wicket_ball"].notna(), df["time_to_event"],
        -(df["max_balls_innings"] - df["total_balls"] + 1))

    cat_cols = ["venue", "batting_team", "bowling_team", "batter", "bowler"]
    features = cat_cols + ["innings", "over", "balls_remaining", "wicket_count", "current_run_rate",
                            "runs_last_6_balls", "wicket_density"]

    train_mask, test_mask = match_level_split(df)
    vocab = fit_native_categories(df.loc[train_mask], cat_cols)
    X = apply_native_categories(df[features], cat_cols, vocab)
    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = df.loc[train_mask, "survival_target"], df.loc[test_mask, "survival_target"]

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
    params = {"objective": "survival:cox", "eval_metric": "cox-nloglik",
              "learning_rate": 0.05, "max_depth": 6, "seed": 42}
    bst = xgb.train(params, dtrain, num_boost_round=200)
    hazard = bst.predict(dtest)

    actual_wicket = (y_test.values > 0).astype(int)
    predicted_class = (hazard > 1.0).astype(int)
    metrics = {"pct_flagged_high_risk": float(predicted_class.mean()),
               "pct_actually_dismissed": float(actual_wicket.mean())}
    print_result(metrics)

    joblib.dump(bst, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_json(vocab, os.path.join(mod_dir, f"{mid}_categories.json"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    results = df.loc[test_mask, ["match_id", "innings", "over", "ball_in_over"]].copy()
    results["actual_wicket"] = actual_wicket
    results["b2_predicted_class"] = predicted_class
    results["b2_predicted_risk"] = hazard
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "batting", "label": name, "column": "b2_predicted_risk",
                     "type": "raw_signed", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics}


# ---------------------------------------------------------------------------
# B3 : Shot Aggression (CatBoost multi-class) - Group A
# ---------------------------------------------------------------------------
def train_b3(df):
    mid, name = "B3", "Shot Aggression"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)

    df = df.copy()
    df["shot_aggression_class"] = np.select(
        [(df["total_runs"] == 0), (df["total_runs"].isin([1, 2, 3])), (df["total_runs"].isin([4, 5, 6]))],
        [0, 1, 2], default=0)
    df["total_balls"] = df["over"] * 6 + df["ball_in_over"]
    df["balls_remaining"] = 120 - df["total_balls"]
    df["wicket_count"] = df.groupby(["match_id", "innings"])["is_wicket"].cumsum()
    df["cumulative_runs"] = df.groupby(["match_id", "innings"])["total_runs"].cumsum()
    df["current_run_rate"] = np.where(df["total_balls"] > 0, (df["cumulative_runs"] / df["total_balls"]) * 6, 0.0)

    cat_cols = ["venue", "batting_team", "bowling_team", "batter", "bowler"]
    features = cat_cols + ["innings", "over", "balls_remaining", "wicket_count", "current_run_rate"]
    for c in cat_cols:
        df[c] = df[c].astype(str)

    train_mask, test_mask = match_level_split(df)
    X_train = df.loc[train_mask, features]
    X_test = df.loc[test_mask, features]
    y_train = df.loc[train_mask, "shot_aggression_class"]
    y_test = df.loc[test_mask, "shot_aggression_class"]

    cat_idx = [features.index(c) for c in cat_cols]
    model = CatBoostClassifier(iterations=500, loss_function="MultiClass", learning_rate=0.05,
                                depth=6, random_seed=42, verbose=0, auto_class_weights="Balanced")
    model.fit(X_train, y_train, cat_features=cat_idx)
    pred = model.predict(X_test).flatten()
    acc = float((pred == y_test.values).mean())
    metrics = {"Accuracy": acc}
    print_result(metrics)

    joblib.dump(model, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_json({"categorical_features": cat_cols}, os.path.join(mod_dir, f"{mid}_categories.json"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    results = df.loc[test_mask, ["match_id", "innings", "over", "ball_in_over"]].copy()
    results["b3_actual_class"] = y_test.values
    results["b3_predicted_class"] = pred
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "batting", "label": name, "column": "b3_predicted_class",
                     "type": "class_0_1_2", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics}


# ---------------------------------------------------------------------------
# B5 : Strike Rotation (Logistic Regression) - Group B
# ---------------------------------------------------------------------------
def train_b5(df):
    mid, name = "B5", "Strike Rotation"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)
    print("  FIX: removed is_boundary / is_dot_ball (both directly ruled the")
    print("  [1,3,5]-runs target in or out for the SAME ball - leakage).")

    df = df.copy()
    df["strike_rotation"] = df["batter_runs"].isin([1, 3, 5]).astype(int)
    df["current_score"] = df.groupby(["match_id", "innings"])["total_runs"].cumsum()
    df["wickets_lost"] = df.groupby(["match_id", "innings"])["is_wicket"].cumsum()
    df["balls_bowled"] = df.groupby(["match_id", "innings"]).cumcount() + 1
    df["balls_remaining"] = 120 - df["balls_bowled"]
    df["current_run_rate"] = df["current_score"] / (df["balls_bowled"] / 6)
    df["wickets_remaining"] = 10 - df["wickets_lost"]
    df["is_boundary"] = df["batter_runs"].isin([4, 6]).astype(int)  # kept only to build PAST momentum below
    df["is_dot_ball"] = (df["total_runs"] == 0).astype(int)
    df["previous_runs"] = df.groupby(["match_id", "innings"])["total_runs"].shift(1).fillna(0)
    df["recent_runs"] = df.groupby(["match_id", "innings"])["total_runs"].rolling(6, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
    df["recent_wickets"] = df.groupby(["match_id", "innings"])["is_wicket"].rolling(12, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
    df["dot_streak"] = df.groupby(["match_id", "innings"])["is_dot_ball"].rolling(3, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
    df["boundary_momentum"] = df.groupby(["match_id", "innings"])["is_boundary"].rolling(6, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
    df["pressure_index"] = df["balls_remaining"] / (df["wickets_remaining"] + 1)
    df["momentum"] = df["recent_runs"] - (df["recent_wickets"] * 6)
    df = add_batter_profile_cols(df, col="batter", prefix="batter")

    features = ["over", "ball_in_over", "current_score", "current_run_rate", "balls_remaining",
                "wickets_lost", "wickets_remaining", "recent_runs", "recent_wickets", "previous_runs",
                "dot_streak", "boundary_momentum", "pressure_index", "momentum",
                "is_powerplay", "is_middle_overs", "is_death_overs",
                "batter_career_strike_rate", "batter_career_dismissal_rate", "batter_career_boundary_pct"]

    train_mask, test_mask = match_level_split(df)
    X_train = df.loc[train_mask, features].fillna(0)
    X_test = df.loc[test_mask, features].fillna(0)
    y_train, y_test = df.loc[train_mask, "strike_rotation"], df.loc[test_mask, "strike_rotation"]

    model = LogisticRegression(solver="lbfgs", class_weight="balanced", C=2.0, max_iter=3000, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, pred, prob)
    print_result(metrics)

    joblib.dump(model, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_json({"features": features}, os.path.join(mod_dir, f"{mid}_features.json"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    results = df.loc[test_mask, ["match_id", "innings", "over", "ball_in_over"]].copy()
    results["b5_actual"] = y_test.values
    results["b5_predicted_prob"] = prob
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "batting", "label": name, "column": "b5_predicted_prob",
                     "type": "probability", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics}


# ---------------------------------------------------------------------------
# B6 : Partnership Stability (Random Forest) - Group B (batter + non_striker)
# ---------------------------------------------------------------------------
def train_b6(df):
    mid, name = "B6", "Partnership Stability"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)
    print("  FIX: switched from a random row-level split to match_level_split -")
    print("  the same partnership's rows were previously able to appear in")
    print("  both train and test.")

    df = df.copy()
    df["partnership_id"] = df["batter"] + "-" + df["non_striker"]
    df["partnership_runs"] = df.groupby(["match_id", "innings", "partnership_id"])["batter_runs"].cumsum()
    target_df = df.groupby(["match_id", "innings", "partnership_id"])["batter_runs"].sum().reset_index()
    target_df.rename(columns={"batter_runs": "final_partnership_score"}, inplace=True)
    df = df.merge(target_df, on=["match_id", "innings", "partnership_id"])
    df = add_batter_profile_cols(df, col="batter", prefix="batter")
    df = add_batter_profile_cols(df, col="non_striker", prefix="non_striker")

    features = ["over", "is_powerplay", "is_middle_overs", "is_death_overs", "partnership_runs",
                "batter_career_strike_rate", "batter_career_dismissal_rate", "batter_career_boundary_pct",
                "non_striker_career_strike_rate", "non_striker_career_dismissal_rate",
                "non_striker_career_boundary_pct"]
    train_mask, test_mask = match_level_split(df)
    X_train = df.loc[train_mask, features].fillna(0)
    X_test = df.loc[test_mask, features].fillna(0)
    y_train, y_test = df.loc[train_mask, "final_partnership_score"], df.loc[test_mask, "final_partnership_score"]

    model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = regression_metrics(y_test, pred)
    print_result(metrics)

    joblib.dump(model, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    save_feature_importance(features, model.feature_importances_, os.path.join(out_dir, f"{mid}_feature_importance.csv"))
    results = df.loc[test_mask, ["match_id", "innings", "over"]].copy()
    results["b6_actual_score"] = y_test.values
    results["b6_predicted_score"] = pred
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "batting", "label": name, "column": "b6_predicted_score",
                     "type": "regression", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics}


# ---------------------------------------------------------------------------
# B8 : Powerplay Exploitation (CatBoost) - unchanged (no player features per spec)
# ---------------------------------------------------------------------------
def train_b8(df):
    mid, name = "B8", "Powerplay Exploitation"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)
    print("  FIX: original had NO train/test split (fit and 'evaluated' on the")
    print("  same rows) - added match_level_split so the reported metric means")
    print("  something.")

    pp_df = df[df["is_powerplay"] == 1].copy()
    pp_df["is_boundary"] = pp_df["batter_runs"].isin([4, 6]).astype(int)
    features = ["over", "ball_in_over", "is_wicket"]

    train_mask, test_mask = match_level_split(pp_df)
    X_train = pp_df.loc[train_mask, features].fillna(0)
    X_test = pp_df.loc[test_mask, features].fillna(0)
    y_train, y_test = pp_df.loc[train_mask, "is_boundary"], pp_df.loc[test_mask, "is_boundary"]

    model = CatBoostClassifier(iterations=200, verbose=0, random_seed=42, auto_class_weights="Balanced")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, pred, prob)
    print_result(metrics)

    joblib.dump(model, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    results = pp_df.loc[test_mask, ["match_id", "innings", "over", "ball_in_over"]].copy()
    results["b8_actual"] = y_test.values
    results["b8_predicted_prob"] = prob
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "batting", "label": name, "column": "b8_predicted_prob",
                     "type": "probability", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics}


# ---------------------------------------------------------------------------
# B9 : Spin Vulnerability (CatBoost) - Group B (batter + partial bowler)
# ---------------------------------------------------------------------------
def train_b9(df):
    mid, name = "B9", "Spin Vulnerability"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)
    print("  FIX: same as B8 - added match_level_split, previously trained and")
    print("  evaluated on the same rows.")
    print("  NOTE: dataset has no bowler-type field, so this is phase-based")
    print("  wicket probability generally, not spin-specific.")

    df2 = df.copy()
    df2 = add_batter_profile_cols(df2, col="batter", prefix="batter")
    df2 = add_bowler_profile_cols_partial(df2, col="bowler")
    features = ["over", "ball_in_over", "is_powerplay", "is_middle_overs", "is_death_overs",
                "batter_career_strike_rate", "batter_career_dismissal_rate", "batter_career_boundary_pct",
                "bowler_career_economy", "bowler_career_dot_pct"]
    train_mask, test_mask = match_level_split(df2)
    X_train = df2.loc[train_mask, features].fillna(0)
    X_test = df2.loc[test_mask, features].fillna(0)
    y_train = df2.loc[train_mask, "is_wicket"].fillna(0).astype(int)
    y_test = df2.loc[test_mask, "is_wicket"].fillna(0).astype(int)

    model = CatBoostClassifier(iterations=200, verbose=0, random_seed=42, auto_class_weights="Balanced")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, pred, prob)
    print_result(metrics)

    joblib.dump(model, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    results = df2.loc[test_mask, ["match_id", "innings", "over", "ball_in_over"]].copy()
    results["b9_actual"] = y_test.values
    results["b9_predicted_prob"] = prob
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "batting", "label": name, "column": "b9_predicted_prob",
                     "type": "probability", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics,
                     "note": "Dataset has no bowler-type (pace/spin) field - this measures general "
                             "phase-based wicket probability, not spin-specific vulnerability."}


# ---------------------------------------------------------------------------
# B10 : Scoring Velocity (LightGBM) - Group A
# ---------------------------------------------------------------------------
def train_b10(df):
    mid, name = "B10", "Scoring Velocity"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)
    print("  FIX: switched from a sequential (shuffle=False) split to")
    print("  match_level_split.")

    df = df.copy()

    def next6_transform(s):
        r = s.to_numpy()
        c = np.cumsum(r)
        n = len(r)
        out = np.full(n, np.nan)
        if n > 6:
            out[: n - 6] = c[6:] - c[: n - 6]
        return pd.Series(out, index=s.index)

    df["next_6_runs"] = df.groupby(["match_id", "innings"])["total_runs"].transform(next6_transform)
    df = df.dropna(subset=["next_6_runs"])

    df["current_score"] = df.groupby(["match_id", "innings"])["total_runs"].cumsum()
    df["wickets_fallen"] = df.groupby(["match_id", "innings"])["is_wicket"].cumsum()
    df["balls_bowled"] = df.groupby(["match_id", "innings"]).cumcount() + 1
    df["current_run_rate"] = df["current_score"] / (df["balls_bowled"] / 6)
    df["balls_remaining"] = 120 - df["balls_bowled"]
    df["wickets_remaining"] = 10 - df["wickets_fallen"]
    df["is_boundary"] = df["batter_runs"].isin([4, 6]).astype(int)
    df["is_dot_ball"] = (df["total_runs"] == 0).astype(int)

    numeric_features = ["over", "ball_in_over", "current_score", "current_run_rate", "balls_bowled",
                         "balls_remaining", "wickets_fallen", "wickets_remaining", "batter_runs", "extra_runs",
                         "total_runs", "is_boundary", "is_dot_ball", "is_powerplay", "is_middle_overs", "is_death_overs"]
    cat_cols = ["batter", "bowler"]
    features = numeric_features + cat_cols

    train_mask, test_mask = match_level_split(df)
    vocab = fit_native_categories(df.loc[train_mask], cat_cols)
    X_all = apply_native_categories(df[features], cat_cols, vocab)
    X_all[numeric_features] = X_all[numeric_features].fillna(0)
    X_train, X_test = X_all.loc[train_mask], X_all.loc[test_mask]
    y_train, y_test = df.loc[train_mask, "next_6_runs"], df.loc[test_mask, "next_6_runs"]

    model = LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=8, num_leaves=31,
                           subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
    model.fit(X_train, y_train, categorical_feature=cat_cols)
    pred = model.predict(X_test)
    metrics = regression_metrics(y_test, pred)
    print_result(metrics)

    joblib.dump(model, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_json(vocab, os.path.join(mod_dir, f"{mid}_categories.json"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    results = df.loc[test_mask, ["match_id", "innings", "over", "ball_in_over"]].copy()
    results["b10_actual_next6"] = y_test.values
    results["b10_predicted_next6"] = pred
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "batting", "label": name, "column": "b10_predicted_next6",
                     "type": "regression", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics}


# ---------------------------------------------------------------------------
# B11 : Targeted Run-Rate (LightGBM) - Group A
# ---------------------------------------------------------------------------
def train_b11(df):
    mid, name = "B11", "Targeted Run-Rate"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)
    print("  FIX: switched from a sequential (shuffle=False) split to")
    print("  match_level_split.")

    df = df.groupby(["match_id", "innings", "over"], as_index=False, group_keys=False).tail(1).reset_index(drop=True)
    df["current_score"] = df.groupby(["match_id", "innings"])["total_runs"].cumsum()
    df["wickets_fallen"] = df.groupby(["match_id", "innings"])["is_wicket"].cumsum()
    df["balls_bowled"] = (df["over"] + 1) * 6
    df["balls_remaining"] = 120 - df["balls_bowled"]
    df["current_run_rate"] = df["current_score"] / (df["balls_bowled"] / 6)
    df["wickets_remaining"] = 10 - df["wickets_fallen"]
    final_score = df.groupby(["match_id", "innings"])["current_score"].max().reset_index().rename(
        columns={"current_score": "final_score"})
    df = df.merge(final_score, on=["match_id", "innings"], how="left")

    numeric_features = ["over", "current_score", "current_run_rate", "balls_remaining", "wickets_fallen",
                         "wickets_remaining", "is_powerplay", "is_middle_overs", "is_death_overs"]
    cat_cols = ["batter", "bowler"]
    features = numeric_features + cat_cols

    train_mask, test_mask = match_level_split(df)
    vocab = fit_native_categories(df.loc[train_mask], cat_cols)
    X_all = apply_native_categories(df[features], cat_cols, vocab)
    X_all[numeric_features] = X_all[numeric_features].fillna(0)
    X_train, X_test = X_all.loc[train_mask], X_all.loc[test_mask]
    y_train, y_test = df.loc[train_mask, "final_score"], df.loc[test_mask, "final_score"]

    model = LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=8, num_leaves=31,
                           subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
    model.fit(X_train, y_train, categorical_feature=cat_cols)
    pred = model.predict(X_test)
    metrics = regression_metrics(y_test, pred)
    print_result(metrics)

    joblib.dump(model, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_json(vocab, os.path.join(mod_dir, f"{mid}_categories.json"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    results = df.loc[test_mask, ["match_id", "innings", "over"]].copy()
    results["b11_actual_final_score"] = y_test.values
    results["b11_predicted_final_score"] = pred
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "batting", "label": name, "column": "b11_predicted_final_score",
                     "type": "regression", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics}


# ---------------------------------------------------------------------------
# B12 : Gap Analysis (KNN) - Group B
# ---------------------------------------------------------------------------
def train_b12(df):
    mid, name = "B12", "Gap Analysis"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)
    print("  FIX: original fit KNN on the first 30k rows then predicted on the")
    print("  WHOLE dataset (including those same 30k rows). Now fits on a")
    print("  train-match sample and evaluates only on held-out test matches.")

    df2 = df.copy()
    df2["is_gap"] = df2["batter_runs"].isin([1, 2, 3]).astype(int)
    df2 = add_batter_profile_cols(df2, col="batter", prefix="batter")
    features = ["over", "ball_in_over", "is_powerplay",
                "batter_career_strike_rate", "batter_career_dismissal_rate", "batter_career_boundary_pct"]

    train_mask, test_mask = match_level_split(df2)
    train_df = df2.loc[train_mask]
    test_df = df2.loc[test_mask]

    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(train_df[features].fillna(0))
    X_test = scaler.transform(test_df[features].fillna(0))

    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X_train_full), size=min(30000, len(X_train_full)), replace=False)
    X_samp = X_train_full[sample_idx]
    y_samp = train_df["is_gap"].to_numpy()[sample_idx]

    model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    model.fit(X_samp, y_samp)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    y_test = test_df["is_gap"].to_numpy()
    metrics = classification_metrics(y_test, pred, prob)
    print_result(metrics)

    joblib.dump({"model": model, "scaler": scaler}, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    results = test_df[["match_id", "innings", "over", "ball_in_over"]].copy()
    results["b12_actual"] = y_test
    results["b12_predicted_prob"] = prob
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "batting", "label": name, "column": "b12_predicted_prob",
                     "type": "probability", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics}


# ---------------------------------------------------------------------------
# B13 : Wicket-Loss Mitigation (Logistic Regression) - Group B
# ---------------------------------------------------------------------------
def train_b13(df):
    mid, name = "B13", "Wicket-Loss Mitigation"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)
    print("  FIX: is_boundary now uses the PREVIOUS ball (was the current ball's")
    print("  own outcome sitting next to is_wicket, also the current ball -")
    print("  leakage) + switched to match_level_split (was sequential).")

    df = df.copy()
    df["current_score"] = df.groupby(["match_id", "innings"])["total_runs"].cumsum()
    df["balls_bowled"] = df.groupby(["match_id", "innings"]).cumcount() + 1
    df["current_run_rate"] = df["current_score"] / (df["balls_bowled"] / 6)
    df["balls_remaining"] = 120 - df["balls_bowled"]
    df["wickets_fallen"] = df.groupby(["match_id", "innings"])["is_wicket"].cumsum()
    df["wickets_remaining"] = 10 - df["wickets_fallen"]
    df["is_boundary_prev"] = df.groupby(["match_id", "innings"])["batter_runs"].shift(1).isin([4, 6]).astype(int)
    df = add_batter_profile_cols(df, col="batter", prefix="batter")

    features = ["over", "ball_in_over", "current_score", "current_run_rate", "balls_remaining",
                "wickets_fallen", "wickets_remaining", "is_boundary_prev",
                "is_powerplay", "is_middle_overs", "is_death_overs",
                "batter_career_strike_rate", "batter_career_dismissal_rate", "batter_career_boundary_pct"]
    train_mask, test_mask = match_level_split(df)
    X_train = df.loc[train_mask, features].fillna(0)
    X_test = df.loc[test_mask, features].fillna(0)
    y_train = df.loc[train_mask, "is_wicket"].fillna(0).astype(int)
    y_test = df.loc[test_mask, "is_wicket"].fillna(0).astype(int)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, pred, prob)
    print_result(metrics)

    joblib.dump(model, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    results = df.loc[test_mask, ["match_id", "innings", "over", "ball_in_over"]].copy()
    results["b13_actual"] = y_test.values
    results["b13_predicted_prob"] = prob
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "batting", "label": name, "column": "b13_predicted_prob",
                     "type": "probability", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics}


# ---------------------------------------------------------------------------
# B14 : Acceleration Capability (LightGBM) - Group A
# ---------------------------------------------------------------------------
def train_b14(df):
    mid, name = "B14", "Acceleration Capability"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)
    print("  FIX: removed is_boundary / is_dot_ball - both are direct functions")
    print("  of batter_runs, which IS the target (this was the R2=0.92 leak).")
    print("  Also switched from a sequential split to match_level_split.")

    death_df = df[df["over"] >= 15].copy()
    death_df["current_score"] = death_df.groupby(["match_id", "innings"])["total_runs"].cumsum()
    death_df["wickets_fallen"] = death_df.groupby(["match_id", "innings"])["is_wicket"].cumsum()
    death_df["balls_bowled"] = death_df.groupby(["match_id", "innings"]).cumcount() + 1
    death_df["current_run_rate"] = death_df["current_score"] / (death_df["balls_bowled"] / 6)
    death_df["balls_remaining"] = 120 - death_df["balls_bowled"]
    death_df["wickets_remaining"] = 10 - death_df["wickets_fallen"]

    numeric_features = ["over", "ball_in_over", "current_score", "current_run_rate", "balls_remaining",
                         "wickets_fallen", "wickets_remaining", "is_powerplay", "is_middle_overs", "is_death_overs"]
    cat_cols = ["batter", "bowler"]
    features = numeric_features + cat_cols

    train_mask, test_mask = match_level_split(death_df)
    vocab = fit_native_categories(death_df.loc[train_mask], cat_cols)
    X_all = apply_native_categories(death_df[features], cat_cols, vocab)
    X_all[numeric_features] = X_all[numeric_features].fillna(0)
    X_train, X_test = X_all.loc[train_mask], X_all.loc[test_mask]
    y_train, y_test = death_df.loc[train_mask, "batter_runs"], death_df.loc[test_mask, "batter_runs"]

    model = LGBMRegressor(objective="regression", n_estimators=500, learning_rate=0.03, max_depth=8,
                           num_leaves=31, random_state=42, verbosity=-1)
    model.fit(X_train, y_train, categorical_feature=cat_cols)
    pred = model.predict(X_test)
    metrics = regression_metrics(y_test, pred)
    print_result(metrics)

    joblib.dump(model, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_json(vocab, os.path.join(mod_dir, f"{mid}_categories.json"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    save_feature_importance(features, model.feature_importances_, os.path.join(out_dir, f"{mid}_feature_importance.csv"))
    results = death_df.loc[test_mask, ["match_id", "innings", "over", "ball_in_over"]].copy()
    results["b14_actual_runs"] = y_test.values
    results["b14_predicted_runs"] = pred
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "batting", "label": name, "column": "b14_predicted_runs",
                     "type": "regression", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics}


# ---------------------------------------------------------------------------
# B15 : Death-Over Optimization (XGBoost) - Group A, mirrors W3's pattern
# ---------------------------------------------------------------------------
def train_b15(df):
    mid, name = "B15", "Death-Over Optimization"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)
    print("  FIX: removed is_boundary / is_dot_ball - the target includes the")
    print("  CURRENT ball's own runs, so these leaked part of it directly.")
    print("  Also switched from a sequential split to match_level_split.")

    df = df.copy()
    df["current_score"] = df.groupby(["match_id", "innings"])["total_runs"].cumsum()
    df["wickets_lost"] = df.groupby(["match_id", "innings"])["is_wicket"].cumsum()
    df["balls_bowled"] = df.groupby(["match_id", "innings"]).cumcount() + 1
    df["balls_remaining"] = 120 - df["balls_bowled"]
    df["current_run_rate"] = df["current_score"] / (df["balls_bowled"] / 6)
    df["wickets_remaining"] = 10 - df["wickets_lost"]
    df["is_boundary"] = df["batter_runs"].isin([4, 6]).astype(int)  # kept only for rolling boundary_count below
    df["recent_runs"] = df.groupby(["match_id", "innings"])["total_runs"].rolling(6, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
    df["recent_wickets"] = df.groupby(["match_id", "innings"])["is_wicket"].rolling(12, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
    df["boundary_count"] = df.groupby(["match_id", "innings"])["is_boundary"].rolling(12, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
    df["momentum"] = df["recent_runs"] - (df["recent_wickets"] * 6)

    death = df[df["over"] >= 15].copy()
    death["remaining_death_runs"] = death.groupby(["match_id", "innings"])["total_runs"].transform(
        lambda x: x[::-1].cumsum()[::-1])

    numeric_features = ["over", "ball_in_over", "current_score", "current_run_rate", "balls_remaining",
                         "wickets_lost", "wickets_remaining", "recent_runs", "recent_wickets",
                         "boundary_count", "momentum"]
    cat_cols = ["batter", "bowler"]
    features = numeric_features + cat_cols

    train_mask, test_mask = match_level_split(death)
    vocab = fit_native_categories(death.loc[train_mask], cat_cols)
    X_all = apply_native_categories(death[features], cat_cols, vocab)
    X_all[numeric_features] = X_all[numeric_features].fillna(0)
    X_train, X_test = X_all.loc[train_mask], X_all.loc[test_mask]
    y_train, y_test = death.loc[train_mask, "remaining_death_runs"], death.loc[test_mask, "remaining_death_runs"]

    model = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=8, subsample=0.8,
                          colsample_bytree=0.8, objective="reg:squarederror", enable_categorical=True,
                          tree_method="hist", random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = regression_metrics(y_test, pred)
    print_result(metrics)

    joblib.dump(model, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_json(vocab, os.path.join(mod_dir, f"{mid}_categories.json"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    results = death.loc[test_mask, ["match_id", "innings", "over", "ball_in_over"]].copy()
    results["b15_actual_remaining"] = y_test.values
    results["b15_predicted_remaining"] = pred
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "batting", "label": name, "column": "b15_predicted_remaining",
                     "type": "regression", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics}


def main():
    df = load_dataset(DATA_PATH)
    print(f"Loaded dataset: {len(df):,} rows across {df['match_id'].nunique():,} matches\n")

    for fn in [train_b1, train_b2, train_b3, train_b5, train_b6, train_b8, train_b9,
               train_b10, train_b11, train_b12, train_b13, train_b14, train_b15]:
        fn(df)
        print()

    ensure_dirs(OUT_ROOT)
    save_json(RESULTS, os.path.join(OUT_ROOT, "..", "batting_live_training_summary.json"))
    print("All batting_live models trained. Summary written to batting_live_training_summary.json")


if __name__ == "__main__":
    main()
