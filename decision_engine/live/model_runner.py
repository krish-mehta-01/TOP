"""
model_runner.py
----------------
Loads every retrained model once (bowling + batting) and exposes
`predict_all(role, state) -> {model_id: raw_scalar}` - the same
`raw_model_outputs` dict shape DecisionEngine.decide() expects.

Each model's feature vector is built to EXACTLY match what it was trained
on (same column order, same categorical encoding) - see training/train_
bowling.py and training/train_batting.py for the source of truth these
lists were copied from.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

_HERE = os.path.dirname(__file__)
MODELS_ROOT = os.path.join(_HERE, "..", "..", "models")

# Which bowling model directory to load from. Defaults to "bowling" (paper 1's
# timeout-snapshot-trained W-series), unchanged unless explicitly overridden -
# set TOP_BOWLING_MODEL_DIR=bowling_ballbyball to serve the ball-by-ball
# -retrained suite (paper 2) instead. Batting is unaffected either way, since
# it is not retrained between the two papers.
BOWLING_MODEL_DIR = os.environ.get("TOP_BOWLING_MODEL_DIR", "bowling")

BOWLING_SNAPSHOT_FEATURES = [
    "venue", "batting_team", "bowling_team", "innings", "over",
    "current_runs_conceded", "current_balls_bowled", "current_wickets",
    "current_economy", "dot_ball_percentage", "boundary_percentage", "match_phase",
]
BOWLING_CATEGORICAL = ["venue", "batting_team", "bowling_team", "innings", "match_phase"]

# Plain-numeric-only feature orders for models with no categorical columns
# (must match the `features = [...]` list in training/train_batting.py exactly).
B6_FEATURES = ["over", "is_powerplay", "is_middle_overs", "is_death_overs", "partnership_runs"]
B8_FEATURES = ["over", "ball_in_over", "is_wicket"]
B9_FEATURES = ["over", "ball_in_over", "is_powerplay", "is_middle_overs", "is_death_overs"]
B10_FEATURES = ["over", "ball_in_over", "current_score", "current_run_rate", "balls_bowled",
                "balls_remaining", "wickets_fallen", "wickets_remaining", "batter_runs", "extra_runs",
                "total_runs", "is_boundary", "is_dot_ball", "is_powerplay", "is_middle_overs", "is_death_overs"]
B11_FEATURES = ["over", "current_score", "current_run_rate", "balls_remaining", "wickets_fallen",
                "wickets_remaining", "is_powerplay", "is_middle_overs", "is_death_overs"]
B12_FEATURES = ["over", "ball_in_over", "is_powerplay"]
B13_FEATURES = ["over", "ball_in_over", "current_score", "current_run_rate", "balls_remaining",
                "wickets_fallen", "wickets_remaining", "is_boundary_prev", "is_powerplay",
                "is_middle_overs", "is_death_overs"]
B14_FEATURES = ["over", "ball_in_over", "current_score", "current_run_rate", "balls_remaining",
                "wickets_fallen", "wickets_remaining", "is_powerplay", "is_middle_overs", "is_death_overs"]
B15_FEATURES = ["over", "ball_in_over", "current_score", "current_run_rate", "balls_remaining",
                "wickets_lost", "wickets_remaining", "recent_runs", "recent_wickets",
                "boundary_count", "momentum"]


def _load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


class ModelRunner:
    def __init__(self):
        self.models = {}
        self.manifests = {}
        self._load_role(BOWLING_MODEL_DIR, [f"W{i}" for i in range(1, 16)])
        self._load_role("batting", ["B1", "B2", "B3", "B5", "B6", "B8", "B9", "B10", "B11",
                                     "B12", "B13", "B14", "B15"])

    def _load_role(self, role, model_ids):
        for mid in model_ids:
            model_dir = os.path.join(MODELS_ROOT, role, mid)
            model_path = os.path.join(model_dir, f"{mid}_model.pkl")
            if not os.path.exists(model_path):
                continue
            self.models[mid] = joblib.load(model_path)
            manifest = {}
            for suffix in ("categories", "encoding", "features"):
                data = _load_json(os.path.join(model_dir, f"{mid}_{suffix}.json"))
                if data is not None:
                    manifest[suffix] = data
            self.manifests[mid] = manifest

    # -- feature vector builders -------------------------------------------------
    def _native_cat_row(self, state: dict, feature_order: list, cat_cols: list, vocab: dict) -> pd.DataFrame:
        row = {}
        for c in feature_order:
            if c in cat_cols:
                val = str(state.get(c, "Unknown"))
                cats = vocab.get(c, [val])
                row[c] = val if val in cats else cats[0]
            else:
                row[c] = state.get(c, 0)
        df = pd.DataFrame([row])[feature_order]
        for c in cat_cols:
            df[c] = pd.Categorical(df[c], categories=vocab.get(c, [df[c].iloc[0]]))
        return df

    def _onehot_row(self, state: dict, numeric: list, categorical: list, schema: list) -> pd.DataFrame:
        num_vals = {c: state.get(c, 0) for c in numeric}
        cat_vals = {c: str(state.get(c, "Unknown")) for c in categorical}
        onehot = {col: 0 for col in schema}
        for c in categorical:
            key = f"{c}_{cat_vals[c]}"
            if key in onehot:
                onehot[key] = 1
        row = {**num_vals, **onehot}
        return pd.DataFrame([row])[numeric + schema]

    def _plain_row(self, state: dict, feature_order: list) -> pd.DataFrame:
        return pd.DataFrame([{c: state.get(c, 0) for c in feature_order}])[feature_order]

    def _catboost_row(self, state: dict, feature_order: list, cat_cols: list):
        row = {c: (str(state.get(c, "Unknown")) if c in cat_cols else state.get(c, 0)) for c in feature_order}
        return pd.DataFrame([row])[feature_order]

    # -- per-model prediction -----------------------------------------------------
    def _predict_bowling_snapshot(self, mid, state, algo_kind):
        manifest = self.manifests[mid]
        model = self.models[mid]
        if algo_kind == "native":
            X = self._native_cat_row(state, BOWLING_SNAPSHOT_FEATURES, BOWLING_CATEGORICAL, manifest["categories"])
        elif algo_kind == "catboost":
            X = self._catboost_row(state, BOWLING_SNAPSHOT_FEATURES, BOWLING_CATEGORICAL)
        elif algo_kind == "onehot":
            enc = manifest["encoding"]
            X = self._onehot_row(state, enc["numeric"], enc["categorical"], enc["onehot_schema"])
        else:
            raise ValueError(algo_kind)

        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(X)[:, 1][0])
        return float(model.predict(X)[0])

    def predict_bowling(self, state: dict) -> dict:
        out = {}
        native_reg = {"W1", "W3", "W11", "W14"}
        native_cls = {"W9", "W15"}
        catboost_ids = {"W2", "W8"}
        onehot_ids = {"W4", "W6", "W7", "W13"}

        for mid in native_reg | native_cls:
            out[mid] = self._predict_bowling_snapshot(mid, state, "native")
        for mid in catboost_ids:
            out[mid] = self._predict_bowling_snapshot(mid, state, "catboost")
        for mid in onehot_ids:
            out[mid] = self._predict_bowling_snapshot(mid, state, "onehot")

        # W5 / W10 : fixed-weight formulas, no model.predict() involved
        out["W5"] = (0.45 * state.get("current_economy", 0) - 2.50 * state.get("current_wickets", 0)
                     - 3.00 * state.get("dot_ball_percentage", 0) + 2.00 * state.get("boundary_percentage", 0))
        field_pressure = state.get("dot_ball_percentage", 0) - state.get("boundary_percentage", 0)
        out["W10"] = ((field_pressure - 2.50 * state.get("current_wickets", 0)
                       - 1.50 * state.get("dot_ball_percentage", 0))
                      - (-field_pressure + 2.50 * state.get("boundary_percentage", 0)))

        # W12 : lookup table
        lookup_obj = self.models["W12"]
        bowler = state.get("bowler")
        profile = lookup_obj["lookup"].get(bowler, lookup_obj["fallback"])
        out["W12"] = profile["career_economy"]

        return out

    def predict_batting(self, state: dict) -> dict:
        out = {}

        # B1 : native cat (venue, batting_team, bowling_team, innings)
        b1_features = ["venue", "batting_team", "bowling_team", "innings", "over",
                       "balls_remaining", "wicket_count", "current_run_rate", "runs_last_6_balls"]
        vocab = self.manifests["B1"]["categories"]
        X = self._native_cat_row(state, b1_features, ["venue", "batting_team", "bowling_team", "innings"], vocab)
        out["B1"] = float(self.models["B1"].predict(X)[0])

        # B2 : xgboost Booster (Cox survival) - needs a DMatrix
        import xgboost as xgb
        b2_features = ["venue", "batting_team", "bowling_team", "innings", "over", "balls_remaining",
                        "wicket_count", "current_run_rate", "runs_last_6_balls", "wicket_density"]
        vocab2 = self.manifests["B2"]["categories"]
        X2 = self._native_cat_row(state, b2_features, ["venue", "batting_team", "bowling_team"], vocab2)
        dmat = xgb.DMatrix(X2, enable_categorical=True)
        out["B2"] = float(self.models["B2"].predict(dmat)[0])

        # B3 : CatBoost multi-class -> probability of class 2 ("attack")
        b3_features = ["venue", "batting_team", "bowling_team", "innings", "over",
                        "balls_remaining", "wicket_count", "current_run_rate"]
        cat_cols3 = self.manifests["B3"]["categories"]["categorical_features"]
        X3 = self._catboost_row(state, b3_features, cat_cols3)
        probs = self.models["B3"].predict_proba(X3)[0]
        out["B3"] = float(probs[-1])  # P(attack)

        # B5 : plain LogisticRegression, all-numeric features
        b5_features = self.manifests["B5"]["features"]["features"]
        X5 = self._plain_row(state, b5_features)
        out["B5"] = float(self.models["B5"].predict_proba(X5)[:, 1][0])

        # Plain-numeric models
        plain_specs = [
            ("B6", B6_FEATURES, "regress"), ("B8", B8_FEATURES, "proba"), ("B9", B9_FEATURES, "proba"),
            ("B10", B10_FEATURES, "regress"), ("B11", B11_FEATURES, "regress"),
            ("B13", B13_FEATURES, "proba"), ("B14", B14_FEATURES, "regress"), ("B15", B15_FEATURES, "regress"),
        ]
        for mid, feats, kind in plain_specs:
            X = self._plain_row(state, feats)
            model = self.models[mid]
            if kind == "proba":
                out[mid] = float(model.predict_proba(X)[:, 1][0])
            else:
                out[mid] = float(model.predict(X)[0])

        # B12 : KNN on a saved StandardScaler
        bundle = self.models["B12"]
        Xb = self._plain_row(state, B12_FEATURES)
        X_scaled = bundle["scaler"].transform(Xb)
        out["B12"] = float(bundle["model"].predict_proba(X_scaled)[:, 1][0])

        return out


_runner = None


def get_runner() -> ModelRunner:
    global _runner
    if _runner is None:
        _runner = ModelRunner()
    return _runner
