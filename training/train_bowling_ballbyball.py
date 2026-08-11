"""
train_bowling_ballbyball.py
----------------------------
Retrains all 15 bowling models (W1-W15) on a PER-BALL basis, for the second
("ball-by-ball") paper. This is a deliberately isolated sibling of
train_bowling.py, not a replacement for it: paper 1's TOP system and every
number it reports depend on train_bowling.py's original models
(models/bowling/, data/output_data/) being left completely untouched.

What changes relative to train_bowling.py: every model here uses
bowling_ballbyball_snapshot() instead of common.bowling_timeout_snapshot() -
i.e. training/evaluation rows are drawn from (near-)every delivery in the
dataset instead of only the last ball of the four strategic-timeout overs
(5, 8, 12, 15 zero-indexed). This mirrors how most of the batting-side models
(B1,B2,B3,B5,B6,B9,B10,B12,B13) already train on the full dataset with no
over restriction - see paper 1's methodology audit.

Two exceptions keep a PHASE restriction rather than becoming fully
unrestricted, because the restriction is part of what the tactic means, not
a timeout-snapshot artifact:
  - W9  (Death Over Accuracy)     -> restricted to is_death_overs == 1
  - W15 (Powerplay Containment)   -> restricted to is_powerplay == 1
This exactly parallels how B8 (Powerplay Exploitation) and B14/B15
(death-overs models) are scoped in train_batting.py.

Same algorithms, same hyperparameters, same target definitions, same
match-level (never row-level) split as the original - only the row-selection
filter changes. Outputs go to NEW, isolated paths:
  - models/bowling_ballbyball/
  - data/output_data_ballbyball/bowling/
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(__file__))
from common import (  # noqa: E402
    BOWLING_CATEGORICAL,
    BOWLING_SNAPSHOT_FEATURES,
    add_bowling_state_features,
    add_future_spell_stats,
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

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "data", "traning_dataset", "dataset.csv")
MODEL_ROOT = os.path.join(HERE, "..", "models", "bowling_ballbyball")
OUT_ROOT = os.path.join(HERE, "..", "data", "output_data_ballbyball", "bowling")

RESULTS = {}


def dirs_for(model_id):
    mod = os.path.join(MODEL_ROOT, model_id)
    out = os.path.join(OUT_ROOT, model_id)
    ensure_dirs(mod, out)
    return mod, out


def bowling_ballbyball_snapshot(df, require_future_balls=False):
    """Ball-by-ball counterpart to common.bowling_timeout_snapshot(): every
    delivery is a candidate training/evaluation row instead of only the last
    ball of the four strategic-timeout overs."""
    snap = df.copy()
    if require_future_balls:
        snap = snap[snap["future_balls"] > 0].copy()
    return snap


def _prep_full(df, require_future_balls=True):
    df = add_bowling_state_features(df)
    df = add_future_spell_stats(df)
    return bowling_ballbyball_snapshot(df, require_future_balls=require_future_balls)


def _native_cat_train_test(snap, target_col, extra_filter=None):
    if extra_filter is not None:
        snap = snap[extra_filter(snap)].copy()
    train_mask, test_mask = match_level_split(snap)
    vocab = fit_native_categories(snap.loc[train_mask], BOWLING_CATEGORICAL)
    X_train = apply_native_categories(snap.loc[train_mask, BOWLING_SNAPSHOT_FEATURES], BOWLING_CATEGORICAL, vocab)
    X_test = apply_native_categories(snap.loc[test_mask, BOWLING_SNAPSHOT_FEATURES], BOWLING_CATEGORICAL, vocab)
    y_train = snap.loc[train_mask, target_col]
    y_test = snap.loc[test_mask, target_col]
    return X_train, X_test, y_train, y_test, vocab, snap, train_mask, test_mask


# ---------------------------------------------------------------------------
# W1 : Economy Predictor (LightGBM Regressor)
# ---------------------------------------------------------------------------
def train_w1(df):
    mid, name = "W1", "Economy Predictor"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)

    full = add_bowling_state_features(df)
    full = add_future_spell_stats(full)
    snap = bowling_ballbyball_snapshot(full, require_future_balls=False)
    snap["final_bowler_economy"] = np.where(
        snap["final_balls"] > 0, (snap["final_runs"] * 6) / snap["final_balls"], np.nan
    )
    snap = snap.dropna(subset=["final_bowler_economy"]).copy()

    X_train, X_test, y_train, y_test, vocab, snap, tr, te = _native_cat_train_test(snap, "final_bowler_economy")

    model = LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=8, num_leaves=31,
                           subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
    model.fit(X_train, y_train, categorical_feature=BOWLING_CATEGORICAL)
    pred = model.predict(X_test)
    metrics = regression_metrics(y_test, pred)
    print_result(metrics)

    joblib.dump(model, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_json(vocab, os.path.join(mod_dir, f"{mid}_categories.json"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    save_feature_importance(BOWLING_SNAPSHOT_FEATURES, model.feature_importances_,
                             os.path.join(out_dir, f"{mid}_feature_importance.csv"))
    results = snap.loc[te, ["match_id", "innings", "over", "ball_in_over", "bowler"]].copy()
    results["w1_actual_economy"] = y_test.values
    results["w1_predicted_economy"] = pred
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "bowling", "label": name, "column": "w1_predicted_economy",
                     "type": "regression", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics}


# ---------------------------------------------------------------------------
# W2 : Wicket Probability Predictor (CatBoost Classifier)
# ---------------------------------------------------------------------------
def train_w2(df):
    mid, name = "W2", "Wicket Probability Predictor"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)

    snap = _prep_full(df, require_future_balls=True)
    snap["future_wicket"] = (snap["future_wickets"] > 0).astype(int)

    train_mask, test_mask = match_level_split(snap)
    cat_idx = [BOWLING_SNAPSHOT_FEATURES.index(c) for c in BOWLING_CATEGORICAL]
    X = snap[BOWLING_SNAPSHOT_FEATURES].copy()
    for c in BOWLING_CATEGORICAL:
        X[c] = X[c].astype(str)
    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = snap.loc[train_mask, "future_wicket"], snap.loc[test_mask, "future_wicket"]

    model = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, random_seed=42,
                                verbose=0, auto_class_weights="Balanced")
    model.fit(X_train, y_train, cat_features=cat_idx)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, pred, prob)
    print_result(metrics)

    joblib.dump(model, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_json({"categorical_features": BOWLING_CATEGORICAL}, os.path.join(mod_dir, f"{mid}_categories.json"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    results = snap.loc[test_mask, ["match_id", "innings", "over", "ball_in_over", "bowler"]].copy()
    results["w2_actual_future_wicket"] = y_test.values
    results["w2_predicted_wicket_prob"] = prob
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "bowling", "label": name, "column": "w2_predicted_wicket_prob",
                     "type": "probability", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics}


# ---------------------------------------------------------------------------
# W3 : Dot Ball Pressure (XGBoost Regressor)
# ---------------------------------------------------------------------------
def train_w3(df):
    mid, name = "W3", "Dot Ball Pressure"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)

    snap = _prep_full(df, require_future_balls=False)
    snap = snap.dropna(subset=["final_dot_ball_percentage"]).copy()
    X_train, X_test, y_train, y_test, vocab, snap, tr, te = _native_cat_train_test(
        snap, "final_dot_ball_percentage")

    model = XGBRegressor(n_estimators=400, learning_rate=0.04, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8, enable_categorical=True,
                          tree_method="hist", random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = regression_metrics(y_test, pred)
    print_result(metrics)

    joblib.dump(model, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_json(vocab, os.path.join(mod_dir, f"{mid}_categories.json"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    results = snap.loc[te, ["match_id", "innings", "over", "ball_in_over", "bowler"]].copy()
    results["w3_actual_dot_pct"] = y_test.values
    results["w3_predicted_dot_pct"] = pred
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "bowling", "label": name, "column": "w3_predicted_dot_pct",
                     "type": "regression", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics}


# ---------------------------------------------------------------------------
# W4 : Variation Control (Logistic Regression, one-hot)
# ---------------------------------------------------------------------------
def train_w4(df):
    mid, name = "W4", "Variation Control"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)

    from common import apply_onehot_schema, fit_onehot_schema
    snap = _prep_full(df, require_future_balls=True)
    snap["variation_success"] = (snap["future_economy"] <= snap["current_economy"]).astype(int)

    train_mask, test_mask = match_level_split(snap)
    numeric = [c for c in BOWLING_SNAPSHOT_FEATURES if c not in BOWLING_CATEGORICAL]
    schema = fit_onehot_schema(snap.loc[train_mask], BOWLING_CATEGORICAL)
    X_train = apply_onehot_schema(snap.loc[train_mask], BOWLING_CATEGORICAL, numeric, schema)
    X_test = apply_onehot_schema(snap.loc[test_mask], BOWLING_CATEGORICAL, numeric, schema)
    y_train, y_test = snap.loc[train_mask, "variation_success"], snap.loc[test_mask, "variation_success"]

    model = LogisticRegression(solver="lbfgs", class_weight="balanced", C=1.0, max_iter=3000, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, pred, prob)
    print_result(metrics)

    joblib.dump(model, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_json({"onehot_schema": schema, "numeric": numeric, "categorical": BOWLING_CATEGORICAL},
               os.path.join(mod_dir, f"{mid}_encoding.json"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    results = snap.loc[test_mask, ["match_id", "innings", "over", "ball_in_over", "bowler"]].copy()
    results["w4_actual"] = y_test.values
    results["w4_predicted_prob"] = prob
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "bowling", "label": name, "column": "w4_predicted_prob",
                     "type": "probability", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics}


# ---------------------------------------------------------------------------
# W5 : Bowling Change Optimization - DETERMINISTIC FORMULA (not ML)
# ---------------------------------------------------------------------------
W5_WEIGHTS = {"economy": 0.45, "wickets": -2.50, "dot_ball_percentage": -3.00, "boundary_percentage": 2.00}


def w5_score(current_economy, current_wickets, dot_ball_percentage, boundary_percentage):
    return (W5_WEIGHTS["economy"] * current_economy
            + W5_WEIGHTS["wickets"] * current_wickets
            + W5_WEIGHTS["dot_ball_percentage"] * dot_ball_percentage
            + W5_WEIGHTS["boundary_percentage"] * boundary_percentage)


def train_w5(df):
    mid, name = "W5", "Bowling Change Optimization"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)
    print("  NOTE: fixed-weight scoring formula, not a trained model (same formula as paper 1).")

    snap = _prep_full(df, require_future_balls=False)
    snap = snap[snap["current_balls_bowled"] > 0].copy()
    snap["optimization_score"] = w5_score(
        snap["current_economy"], snap["current_wickets"],
        snap["dot_ball_percentage"], snap["boundary_percentage"])

    model_obj = {"algorithm": "fixed_weight_formula", "weights": W5_WEIGHTS,
                 "note": "score = 0.45*economy - 2.50*wickets - 3.00*dot% + 2.00*boundary%; lower is better"}
    joblib.dump(model_obj, os.path.join(mod_dir, f"{mid}_model.pkl"))

    metrics = {"mean": float(snap["optimization_score"].mean()), "std": float(snap["optimization_score"].std()),
                "n": int(len(snap))}
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    snap[["match_id", "innings", "over", "ball_in_over", "bowler", "optimization_score"]].to_csv(
        os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "bowling", "label": name, "column": "optimization_score",
                     "type": "raw_signed_formula", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics}


# ---------------------------------------------------------------------------
# W6 : Line & Length Consistency (Gaussian Naive Bayes, one-hot)
# ---------------------------------------------------------------------------
def train_w6(df):
    mid, name = "W6", "Line & Length Consistency"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)

    from common import apply_onehot_schema, fit_onehot_schema
    snap = _prep_full(df, require_future_balls=True)
    snap["line_length_consistency"] = (snap["future_boundary_percentage"] <= snap["boundary_percentage"]).astype(int)

    train_mask, test_mask = match_level_split(snap)
    numeric = [c for c in BOWLING_SNAPSHOT_FEATURES if c not in BOWLING_CATEGORICAL]
    schema = fit_onehot_schema(snap.loc[train_mask], BOWLING_CATEGORICAL)
    X_train = apply_onehot_schema(snap.loc[train_mask], BOWLING_CATEGORICAL, numeric, schema)
    X_test = apply_onehot_schema(snap.loc[test_mask], BOWLING_CATEGORICAL, numeric, schema)
    y_train = snap.loc[train_mask, "line_length_consistency"]
    y_test = snap.loc[test_mask, "line_length_consistency"]

    model = GaussianNB()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, pred, prob)
    print_result(metrics)

    joblib.dump(model, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_json({"onehot_schema": schema, "numeric": numeric, "categorical": BOWLING_CATEGORICAL},
               os.path.join(mod_dir, f"{mid}_encoding.json"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    results = snap.loc[test_mask, ["match_id", "innings", "over", "ball_in_over", "bowler"]].copy()
    results["w6_actual"] = y_test.values
    results["w6_predicted_prob"] = prob
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "bowling", "label": name, "column": "w6_predicted_prob",
                     "type": "probability", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics}


# ---------------------------------------------------------------------------
# W7 : Spin Control (Random Forest, one-hot)
# ---------------------------------------------------------------------------
def train_w7(df):
    mid, name = "W7", "Spin Control"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)

    from common import apply_onehot_schema, fit_onehot_schema
    snap = _prep_full(df, require_future_balls=True)
    snap["spin_control_success"] = (snap["future_dot_ball_percentage"] >= snap["dot_ball_percentage"]).astype(int)

    train_mask, test_mask = match_level_split(snap)
    numeric = [c for c in BOWLING_SNAPSHOT_FEATURES if c not in BOWLING_CATEGORICAL]
    schema = fit_onehot_schema(snap.loc[train_mask], BOWLING_CATEGORICAL)
    X_train = apply_onehot_schema(snap.loc[train_mask], BOWLING_CATEGORICAL, numeric, schema)
    X_test = apply_onehot_schema(snap.loc[test_mask], BOWLING_CATEGORICAL, numeric, schema)
    y_train = snap.loc[train_mask, "spin_control_success"]
    y_test = snap.loc[test_mask, "spin_control_success"]

    model = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced",
                                    random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, pred, prob)
    print_result(metrics)

    joblib.dump(model, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_json({"onehot_schema": schema, "numeric": numeric, "categorical": BOWLING_CATEGORICAL},
               os.path.join(mod_dir, f"{mid}_encoding.json"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    results = snap.loc[test_mask, ["match_id", "innings", "over", "ball_in_over", "bowler"]].copy()
    results["w7_actual"] = y_test.values
    results["w7_predicted_prob"] = prob
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "bowling", "label": name, "column": "w7_predicted_prob",
                     "type": "probability", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics,
                     "note": "Dataset has no bowler-type (pace/spin) field, so this measures dot-ball "
                             "control generally - it cannot actually condition on spin bowling specifically."}


# ---------------------------------------------------------------------------
# W8 : Yorker Effectiveness (CatBoost Classifier)
# ---------------------------------------------------------------------------
def train_w8(df):
    mid, name = "W8", "Yorker Effectiveness"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)

    snap = _prep_full(df, require_future_balls=True)
    snap["yorker_effectiveness"] = (snap["future_boundary_percentage"] <= snap["boundary_percentage"]).astype(int)

    train_mask, test_mask = match_level_split(snap)
    cat_idx = [BOWLING_SNAPSHOT_FEATURES.index(c) for c in BOWLING_CATEGORICAL]
    X = snap[BOWLING_SNAPSHOT_FEATURES].copy()
    for c in BOWLING_CATEGORICAL:
        X[c] = X[c].astype(str)
    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train = snap.loc[train_mask, "yorker_effectiveness"]
    y_test = snap.loc[test_mask, "yorker_effectiveness"]

    model = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, random_seed=42,
                                verbose=0, auto_class_weights="Balanced")
    model.fit(X_train, y_train, cat_features=cat_idx)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, pred, prob)
    print_result(metrics)

    joblib.dump(model, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_json({"categorical_features": BOWLING_CATEGORICAL}, os.path.join(mod_dir, f"{mid}_categories.json"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    results = snap.loc[test_mask, ["match_id", "innings", "over", "ball_in_over", "bowler"]].copy()
    results["w8_actual"] = y_test.values
    results["w8_predicted_prob"] = prob
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "bowling", "label": name, "column": "w8_predicted_prob",
                     "type": "probability", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics,
                     "note": "Dataset has no delivery-type field, so this cannot verify yorkers "
                             "specifically - it measures general future boundary-prevention."}


# ---------------------------------------------------------------------------
# W9 : Death Over Accuracy (LightGBM Classifier) - phase-restricted (death overs)
# ---------------------------------------------------------------------------
def train_w9(df):
    mid, name = "W9", "Death Over Accuracy"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)
    print("  NOTE: restricted to is_death_overs==1 (every death-overs ball, not just")
    print("  the over=15 timeout snapshot) - phase-scoping kept, snapshot restriction dropped.")

    snap = _prep_full(df, require_future_balls=True)
    snap = snap[snap["is_death_overs"] == 1].copy()
    snap["death_over_accuracy"] = (
        (snap["future_economy"] <= snap["current_economy"])
        & (snap["future_boundary_percentage"] <= snap["boundary_percentage"])
    ).astype(int)

    X_train, X_test, y_train, y_test, vocab, snap, tr, te = _native_cat_train_test(snap, "death_over_accuracy")

    model = LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
                            max_depth=8, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
    model.fit(X_train, y_train, categorical_feature=BOWLING_CATEGORICAL)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, pred, prob)
    print_result(metrics)

    joblib.dump(model, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_json(vocab, os.path.join(mod_dir, f"{mid}_categories.json"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    results = snap.loc[te, ["match_id", "innings", "over", "ball_in_over", "bowler"]].copy()
    results["w9_actual"] = y_test.values
    results["w9_predicted_prob"] = prob
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "bowling", "label": name, "column": "w9_predicted_prob",
                     "type": "probability", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics,
                     "note": "Differentiated from W15 by phase (is_death_overs==1) only - the dataset "
                             "has no delivery-type field, so this can't model death-over-specific "
                             "tactics beyond 'is the bowler in good economy/boundary form'."}


# ---------------------------------------------------------------------------
# W10 : Field-Set Optimization - DETERMINISTIC FORMULA (not ML)
# ---------------------------------------------------------------------------
def train_w10(df):
    mid, name = "W10", "Field-Set Optimization"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)
    print("  NOTE: fixed-weight scoring formula, not a trained model (same formula as paper 1).")

    snap = _prep_full(df, require_future_balls=False)
    snap = snap[snap["current_balls_bowled"] > 0].copy()
    snap["field_pressure_score"] = snap["dot_ball_percentage"] - snap["boundary_percentage"]
    snap["optimization_score"] = (
        snap["field_pressure_score"] - (2.50 * snap["current_wickets"]) - (1.50 * snap["dot_ball_percentage"])
    ) - (-snap["field_pressure_score"] + (2.50 * snap["boundary_percentage"]))

    model_obj = {"algorithm": "fixed_weight_formula",
                 "note": "optimization_score > 0 favours an attacking field, < 0 favours defensive"}
    joblib.dump(model_obj, os.path.join(mod_dir, f"{mid}_model.pkl"))

    metrics = {"mean": float(snap["optimization_score"].mean()), "std": float(snap["optimization_score"].std()),
                "n": int(len(snap))}
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    snap[["match_id", "innings", "over", "ball_in_over", "bowler", "optimization_score"]].to_csv(
        os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "bowling", "label": name, "column": "optimization_score",
                     "type": "raw_signed_formula", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics}


# ---------------------------------------------------------------------------
# W11 : Run-Containment (LightGBM Regressor)
# ---------------------------------------------------------------------------
def train_w11(df):
    mid, name = "W11", "Run-Containment"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)

    snap = _prep_full(df, require_future_balls=True)
    X_train, X_test, y_train, y_test, vocab, snap, tr, te = _native_cat_train_test(snap, "future_runs")

    model = LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=8, num_leaves=31,
                           subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
    model.fit(X_train, y_train, categorical_feature=BOWLING_CATEGORICAL)
    pred = model.predict(X_test)
    metrics = regression_metrics(y_test, pred)
    print_result(metrics)

    joblib.dump(model, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_json(vocab, os.path.join(mod_dir, f"{mid}_categories.json"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    results = snap.loc[te, ["match_id", "innings", "over", "ball_in_over", "bowler"]].copy()
    results["w11_actual_future_runs"] = y_test.values
    results["w11_predicted_future_runs"] = pred
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "bowling", "label": name, "column": "w11_predicted_future_runs",
                     "type": "regression", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics}


# ---------------------------------------------------------------------------
# W12 : Bowler Form / Baseline - JSON lookup, train-only (leakage-safe)
# ---------------------------------------------------------------------------
def train_w12(df):
    mid, name = "W12", "Bowler Form / Baseline"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)

    full = add_bowling_state_features(df)
    snap = bowling_ballbyball_snapshot(full, require_future_balls=False)
    train_mask, test_mask = match_level_split(snap)
    train_bowlers_df = full[full["match_id"].isin(snap.loc[train_mask, "match_id"].unique())]

    bowler_summary = (
        train_bowlers_df.groupby("bowler")
        .agg(matches=("match_id", "nunique"), total_runs=("runs_conceded", "sum"),
             total_balls=("legal_delivery", "sum"), total_wickets=("is_wicket", "sum"),
             total_dot_balls=("is_dot_ball", "sum"), total_boundaries=("is_boundary", "sum"))
        .reset_index()
    )
    bowler_summary["career_economy"] = np.where(
        bowler_summary["total_balls"] > 0, (bowler_summary["total_runs"] * 6) / bowler_summary["total_balls"], np.nan)
    bowler_summary["dot_ball_percentage"] = np.where(
        bowler_summary["total_balls"] > 0, bowler_summary["total_dot_balls"] / bowler_summary["total_balls"], np.nan)
    bowler_summary["boundary_percentage"] = np.where(
        bowler_summary["total_balls"] > 0, bowler_summary["total_boundaries"] / bowler_summary["total_balls"], np.nan)
    bowler_summary["wickets_per_over"] = np.where(
        bowler_summary["total_balls"] > 0, bowler_summary["total_wickets"] / (bowler_summary["total_balls"] / 6), np.nan)

    global_fallback = {
        "career_economy": float(bowler_summary["career_economy"].mean()),
        "dot_ball_percentage": float(bowler_summary["dot_ball_percentage"].mean()),
        "boundary_percentage": float(bowler_summary["boundary_percentage"].mean()),
        "wickets_per_over": float(bowler_summary["wickets_per_over"].mean()),
    }

    lookup = {
        row["bowler"]: {
            "career_economy": round(row["career_economy"], 3),
            "dot_ball_percentage": round(row["dot_ball_percentage"], 4),
            "boundary_percentage": round(row["boundary_percentage"], 4),
            "wickets_per_over": round(row["wickets_per_over"], 3),
        }
        for _, row in bowler_summary.iterrows()
    }
    joblib.dump({"lookup": lookup, "fallback": global_fallback, "algorithm": "train_only_lookup_table"},
                os.path.join(mod_dir, f"{mid}_model.pkl"))

    full = add_future_spell_stats(full)
    test_snap = bowling_ballbyball_snapshot(full, require_future_balls=True)
    test_snap = test_snap[test_snap["match_id"].isin(snap.loc[test_mask, "match_id"].unique())].copy()
    test_snap["career_economy"] = test_snap["bowler"].map(lambda b: lookup.get(b, global_fallback)["career_economy"])
    valid = test_snap.dropna(subset=["career_economy", "future_economy"])
    corr = float(np.corrcoef(valid["career_economy"], valid["future_economy"])[0, 1]) if len(valid) > 5 else None
    metrics = {"n_bowlers_in_lookup": len(lookup), "n_test_snapshots": len(test_snap),
               "correlation_career_economy_vs_future_economy": corr}
    print_result({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))

    results = test_snap[["match_id", "innings", "over", "ball_in_over", "bowler", "career_economy",
                          "dot_ball_percentage", "boundary_percentage"]].copy()
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "bowling", "label": name, "column": "career_economy",
                     "type": "lookup", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics}


# ---------------------------------------------------------------------------
# W13 : L/R Matchup Bias (Logistic Regression, one-hot)
# ---------------------------------------------------------------------------
def train_w13(df):
    mid, name = "W13", "L/R Matchup Bias"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)

    from common import apply_onehot_schema, fit_onehot_schema
    snap = _prep_full(df, require_future_balls=True)
    snap["matchup_bias"] = (
        (snap["future_economy"] <= snap["current_economy"]) & (snap["future_wickets"] >= 1)
    ).astype(int)

    train_mask, test_mask = match_level_split(snap)
    numeric = [c for c in BOWLING_SNAPSHOT_FEATURES if c not in BOWLING_CATEGORICAL]
    schema = fit_onehot_schema(snap.loc[train_mask], BOWLING_CATEGORICAL)
    X_train = apply_onehot_schema(snap.loc[train_mask], BOWLING_CATEGORICAL, numeric, schema)
    X_test = apply_onehot_schema(snap.loc[test_mask], BOWLING_CATEGORICAL, numeric, schema)
    y_train, y_test = snap.loc[train_mask, "matchup_bias"], snap.loc[test_mask, "matchup_bias"]

    model = LogisticRegression(solver="lbfgs", class_weight="balanced", C=1.0, max_iter=3000, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, pred, prob)
    print_result(metrics)

    joblib.dump(model, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_json({"onehot_schema": schema, "numeric": numeric, "categorical": BOWLING_CATEGORICAL},
               os.path.join(mod_dir, f"{mid}_encoding.json"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    results = snap.loc[test_mask, ["match_id", "innings", "over", "ball_in_over", "bowler"]].copy()
    results["w13_actual"] = y_test.values
    results["w13_predicted_prob"] = prob
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "bowling", "label": name, "column": "w13_predicted_prob",
                     "type": "probability", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics,
                     "note": "Dataset has no batter handedness field, so this measures general "
                             "spell-improvement + wicket-taking bias, not literal left/right-handed matchups."}


# ---------------------------------------------------------------------------
# W14 : Economy Trend Analysis (LightGBM Regressor)
# ---------------------------------------------------------------------------
def train_w14(df):
    mid, name = "W14", "Economy Trend Analysis"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)

    snap = _prep_full(df, require_future_balls=True)
    X_train, X_test, y_train, y_test, vocab, snap, tr, te = _native_cat_train_test(snap, "future_economy")

    model = LGBMRegressor(n_estimators=400, learning_rate=0.04, max_depth=7, num_leaves=31,
                           subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
    model.fit(X_train, y_train, categorical_feature=BOWLING_CATEGORICAL)
    pred = model.predict(X_test)
    metrics = regression_metrics(y_test, pred)
    metrics["prediction_std"] = float(np.std(pred))
    print_result(metrics)

    joblib.dump(model, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_json(vocab, os.path.join(mod_dir, f"{mid}_categories.json"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    results = snap.loc[te, ["match_id", "innings", "over", "ball_in_over", "bowler"]].copy()
    results["w14_actual_future_economy"] = y_test.values
    results["w14_predicted_future_economy"] = pred
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "bowling", "label": name, "column": "w14_predicted_future_economy",
                     "type": "regression", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics}


# ---------------------------------------------------------------------------
# W15 : Powerplay Containment (LightGBM Classifier) - phase-restricted (powerplay)
# ---------------------------------------------------------------------------
def train_w15(df):
    mid, name = "W15", "Powerplay Containment"
    print_header(mid, name)
    mod_dir, out_dir = dirs_for(mid)
    print("  NOTE: restricted to is_powerplay==1 (every powerplay ball, not just the")
    print("  over=5 timeout snapshot) - phase-scoping kept, snapshot restriction dropped.")

    snap = _prep_full(df, require_future_balls=True)
    snap = snap[snap["is_powerplay"] == 1].copy()
    snap["powerplay_containment"] = (
        (snap["future_economy"] <= snap["current_economy"])
        & (snap["future_boundary_percentage"] <= snap["boundary_percentage"])
    ).astype(int)

    X_train, X_test, y_train, y_test, vocab, snap, tr, te = _native_cat_train_test(snap, "powerplay_containment")

    model = LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
                            max_depth=8, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
    model.fit(X_train, y_train, categorical_feature=BOWLING_CATEGORICAL)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, pred, prob)
    print_result(metrics)

    joblib.dump(model, os.path.join(mod_dir, f"{mid}_model.pkl"))
    save_json(vocab, os.path.join(mod_dir, f"{mid}_categories.json"))
    save_metrics(metrics, os.path.join(out_dir, f"{mid}_model_metrics.csv"))
    results = snap.loc[te, ["match_id", "innings", "over", "ball_in_over", "bowler"]].copy()
    results["w15_actual"] = y_test.values
    results["w15_predicted_prob"] = prob
    results.to_csv(os.path.join(out_dir, f"{mid}_results.csv"), index=False)

    RESULTS[mid] = {"role": "bowling", "label": name, "column": "w15_predicted_prob",
                     "type": "probability", "csv": f"{mid}/{mid}_results.csv", "metrics": metrics,
                     "note": "Differentiated from W9 by phase (is_powerplay==1) only - same dataset "
                             "limitation as W9 applies (no delivery-type field)."}


def main():
    df = load_dataset(DATA_PATH)
    print(f"Loaded dataset: {len(df):,} rows across {df['match_id'].nunique():,} matches\n")

    for fn in [train_w1, train_w2, train_w3, train_w4, train_w5, train_w6, train_w7,
               train_w8, train_w9, train_w10, train_w11, train_w12, train_w13, train_w14, train_w15]:
        fn(df)
        print()

    ensure_dirs(OUT_ROOT)
    save_json(RESULTS, os.path.join(OUT_ROOT, "..", "bowling_ballbyball_training_summary.json"))
    print("All ball-by-ball bowling models trained. Summary written to bowling_ballbyball_training_summary.json")


if __name__ == "__main__":
    main()
