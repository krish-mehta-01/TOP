"""
build_reliability_live.py
---------------------------
Turns each "live" model's real validation metric into a reliability
multiplier for the Weighter, and writes decision_engine/config/reliability_live.json.
Directly modeled on training/build_reliability.py - same formula, same
per-type special cases - just reading the live training summaries
(data/output_data_live/bowling_live_training_summary.json and
data/output_data_live/batting_live_training_summary.json, both produced by
train_bowling_live.py / train_batting_live.py which retrain every model,
bowling AND batting) instead of the paper-1 summaries.

Mapping (identical to build_reliability.py):
  classifiers (have ROC_AUC) : reliability = clip(0.4 + 1.2*(AUC-0.5), 0.2, 1.6)
  regressors (have R2)      : reliability = clip(0.2 + max(R2, 0), 0.15, 1.6)
  W12 (lookup table)         : reliability = clip(|correlation| * 2, 0.15, 1.0)
  W5 / W10 (fixed formulas)  : 0.7 flat
  B2 (Cox survival)          : 0.6 flat
  B3 (3-class accuracy)      : reliability scaled against the 0.333 random baseline
"""

import json
import os

OUT_LIVE_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "output_data_live")
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "decision_engine", "config")


def compute_reliability():
    bowling = json.load(open(os.path.join(OUT_LIVE_ROOT, "bowling_live_training_summary.json")))
    batting = json.load(open(os.path.join(OUT_LIVE_ROOT, "batting_live_training_summary.json")))
    all_models = {**bowling, **batting}

    reliability = {}
    for mid, info in all_models.items():
        m = info["metrics"]
        if "ROC_AUC" in m and m["ROC_AUC"] is not None:
            auc = m["ROC_AUC"]
            rel = max(0.2, min(1.6, 0.4 + 1.2 * (auc - 0.5)))
        elif "R2" in m:
            r2 = m["R2"]
            rel = max(0.15, min(1.6, 0.2 + max(r2, 0)))
        elif mid == "W12":
            corr = m.get("correlation_career_economy_vs_future_economy", 0) or 0
            rel = max(0.15, min(1.0, abs(corr) * 2))
        elif mid in ("W5", "W10"):
            rel = 0.7
        elif mid == "B2":
            rel = 0.6
        elif mid == "B3":
            acc = m.get("Accuracy", 0.333)
            rel = max(0.15, min(1.0, (acc - 0.333) / 0.667 * 1.2 + 0.3))
        else:
            rel = 1.0
        reliability[mid] = round(rel, 3)
    return reliability


def main():
    reliability = compute_reliability()
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(os.path.join(CONFIG_DIR, "reliability_live.json"), "w") as f:
        json.dump(reliability, f, indent=2)
    for mid in sorted(reliability, key=lambda x: (x[0], int(x[1:]))):
        print(f"{mid:5s} reliability={reliability[mid]:.3f}")
    print(f"\nWrote {len(reliability)} entries to config/reliability_live.json")


if __name__ == "__main__":
    main()
